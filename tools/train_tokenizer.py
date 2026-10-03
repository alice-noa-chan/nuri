"""Train byte-level BPE variants without copying input provenance into artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = (
    "[PAD]", "[MASK]", "[EOS]", "[SEP]",
    "<|system|>", "<|user|>", "<|assistant|>",
)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] + '[EOS]' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def read_records(
    path: str | Path, file_format: str = "text", max_records: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (text, language); JSONL uses only generic text/lang fields."""
    if file_format not in {"text", "jsonl"}:
        raise ValueError("file_format must be text or jsonl")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    yielded = 0
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if file_format == "jsonl":
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on record {line_number}") from error
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    raise ValueError(f"Record {line_number} needs a string text field")
                text, language = item["text"], item.get("lang", "und")
                if not isinstance(language, str):
                    raise ValueError(f"Record {line_number} needs a string lang field")
            else:
                text, language = line, "und"
            if not text.strip():
                continue
            yield text, language
            yielded += 1
            if max_records is not None and yielded >= max_records:
                break


def evaluate_compression(
    tokenizer: Tokenizer, records: Iterator[tuple[str, str]],
) -> dict:
    totals: dict[str, dict] = defaultdict(lambda: {"records": 0, "bytes": 0, "tokens": 0})
    for text, language in records:
        row = totals[language]
        row["records"] += 1
        row["bytes"] += len(text.encode("utf-8"))
        row["tokens"] += len(tokenizer.encode(text, add_special_tokens=False).ids)
    result = {}
    for language, row in sorted(totals.items()):
        result[language] = dict(row, bytes_per_token=row["bytes"] / max(row["tokens"], 1))
    return result


def train_variant(
    input_file: str | Path,
    output_dir: str | Path,
    vocab_size: int,
    *,
    file_format: str = "text",
    eval_file: str | Path | None = None,
    eval_format: str | None = None,
    min_frequency: int = 2,
    max_records: int | None = None,
    model_max_length: int = 1024,
) -> dict:
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must fit 256 byte symbols and all special tokens")
    if min_frequency < 1 or model_max_length < 1:
        raise ValueError("min_frequency and model_max_length must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "metrics.json"):
        if (output_dir / filename).exists():
            raise FileExistsError("Tokenizer output already exists; choose a new output directory")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    trained_records = 0

    def text_stream():
        nonlocal trained_records
        for text, _ in read_records(input_file, file_format, max_records):
            trained_records += 1
            yield text

    tokenizer.train_from_iterator(text_stream(), trainer=trainer)
    if not trained_records:
        raise ValueError("Training input contains no nonempty text records")
    tokenizer.save(str(output_dir / "tokenizer.json"))
    special_map = {
        "pad_token": SPECIAL_TOKENS[0], "mask_token": SPECIAL_TOKENS[1],
        "eos_token": SPECIAL_TOKENS[2], "sep_token": SPECIAL_TOKENS[3],
        "additional_special_tokens": list(SPECIAL_TOKENS[4:]),
    }
    tokenizer_config = dict(
        special_map,
        tokenizer_class="PreTrainedTokenizerFast",
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
        chat_template=CHAT_TEMPLATE,
    )
    metrics = {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "trained_records": trained_records,
        "evaluation_split": "held_out" if eval_file is not None else "training_input",
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "compression": evaluate_compression(
            tokenizer,
            read_records(eval_file or input_file, eval_format or file_format, max_records),
        ),
    }
    for filename, contents in (
        ("tokenizer_config.json", tokenizer_config),
        ("special_tokens_map.json", special_map),
        ("metrics.json", metrics),
    ):
        (output_dir / filename).write_text(
            json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=os.environ.get("NURI_TOKENIZER_INPUT"))
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--eval-file", default=os.environ.get("NURI_TOKENIZER_EVAL"))
    parser.add_argument("--eval-format", choices=["text", "jsonl"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[32768, 49152, 65536])
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--model-max-length", type=int, default=1024)
    args = parser.parse_args(argv)
    if not args.input_file:
        parser.error("Set --input-file or NURI_TOKENIZER_INPUT")
    if len(args.vocab_sizes) != len(set(args.vocab_sizes)):
        parser.error("--vocab-sizes must not contain duplicate variants")
    results = []
    for vocab_size in args.vocab_sizes:
        metrics = train_variant(
            args.input_file, Path(args.output_dir) / f"vocab-{vocab_size}", vocab_size,
            file_format=args.format, eval_file=args.eval_file, eval_format=args.eval_format,
            min_frequency=args.min_frequency, max_records=args.max_records,
            model_max_length=args.model_max_length,
        )
        results.append(metrics)
    print(json.dumps({"variants": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
