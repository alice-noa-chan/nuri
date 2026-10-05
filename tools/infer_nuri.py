"""Run text generation or embeddings from a local Nuri checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.checkpoint import load_pretrained
from nuri.data import chat_prompt_text
from nuri.generation import generate
from nuri.model import NuriModel


def load_tokenizer(path: str | Path, model: NuriModel) -> Tokenizer:
    path = Path(path)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))
    expected = {"[PAD]": model.config.pad_id, "[MASK]": model.config.mask_id, "[EOS]": model.config.eos_id}
    for token, token_id in expected.items():
        if tokenizer.token_to_id(token) != token_id:
            raise ValueError(f"Tokenizer {token} ID must match model config ({token_id})")
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size must match model config; use the tokenizer trained with this checkpoint")
    return tokenizer


def tokenize_batch(
    tokenizer: Tokenizer,
    model: NuriModel,
    texts: list[str],
    *,
    chat: bool = False,
    truncate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Provide at least one nonempty text; every batch item must contain text")
    rows = []
    for text in texts:
        if chat:
            for token in ("<|user|>", "<|assistant|>"):
                if tokenizer.token_to_id(token) is None:
                    raise ValueError(f"Chat generation requires the {token} token")
            text = chat_prompt_text([{"role": "user", "content": text}])
        row = tokenizer.encode(text, add_special_tokens=False).ids
        if len(row) > model.config.max_input_len:
            if not truncate:
                raise ValueError("Input exceeds max_input_len; shorten it or explicitly pass --truncate")
            row = row[-model.config.max_input_len:] if chat else row[:model.config.max_input_len]
        if not row:
            raise ValueError("Tokenizer produced an empty input")
        rows.append(row)
    device = next(model.parameters()).device
    ids = torch.full((len(rows), max(map(len, rows))), model.config.pad_id, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, device=device)
        valid[index, :len(row)] = True
    return ids, valid


@torch.inference_mode()
def infer(
    model: NuriModel,
    tokenizer: Tokenizer,
    texts: list[str],
    *,
    task: str = "generate",
    chat: bool = False,
    truncate: bool = False,
    steps: int | None = None,
    rounds: int | None = None,
    canvas_len: int | None = None,
    max_len: int | None = None,
    block_size: int | None = None,
    mode: str = "fast",
    dim: int | None = None,
) -> dict:
    if task not in {"generate", "embed"}:
        raise ValueError("task must be generate or embed")
    if chat and task != "generate":
        raise ValueError("Chat formatting applies only to generation")
    if block_size is not None and task != "generate":
        raise ValueError("block_size applies only to generation")
    if steps is not None and steps < 0:
        raise ValueError("Reasoning steps must be nonnegative")
    for name, value in (("rounds", rounds), ("canvas_len", canvas_len), ("max_len", max_len),
                        ("block_size", block_size)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if dim is not None and not 1 <= dim <= model.config.embed_dim:
        raise ValueError("dim must be between 1 and the checkpoint embedding dimension")
    model.eval()
    ids, valid = tokenize_batch(tokenizer, model, texts, chat=chat, truncate=truncate)
    result = {"task": task, "input_tokens": valid.sum(dim=1).tolist()}
    if task == "embed":
        vectors = model.embed(ids, valid, mode=mode, steps=steps, dim=dim)
        result.update(mode=mode, dimension=vectors.shape[1], embeddings=vectors.float().cpu().tolist())
    else:
        generated = generate(model, ids, valid, steps=steps, rounds=rounds, canvas_len=canvas_len,
                             max_len=max_len, block_size=block_size)
        result.update(
            answers=tokenizer.decode_batch(generated.tokens, skip_special_tokens=True),
            token_ids=generated.tokens,
            canvas_lengths=generated.lengths.tolist(),
            ended_with_eos=generated.ended_with_eos,
            reasoning_steps=model.config.default_depth if steps is None else steps,
            refinement_rounds=model.config.refine_rounds if rounds is None else rounds,
            block_size=block_size,
            readout_calls=generated.readout_calls,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Folder containing config.json and model.safetensors")
    parser.add_argument("--tokenizer", help="Tokenizer file or folder; defaults to checkpoint folder")
    parser.add_argument("--task", choices=["generate", "embed"], default="generate")
    parser.add_argument("--text", action="append", required=True, help="Repeat this flag for a batch")
    parser.add_argument("--chat", action="store_true", help="Format each text as a user turn for an IT checkpoint")
    parser.add_argument("--truncate", action="store_true", help="Keep the last input tokens for chat, first tokens otherwise")
    parser.add_argument("--steps", type=int, help="Reasoning depth T")
    parser.add_argument("--rounds", type=int, help="Refinement rounds S per block (whole canvas by default)")
    parser.add_argument("--block-size", type=int, help="Opt-in left-to-right block width; earlier blocks stay fixed")
    parser.add_argument("--canvas-len", type=int, help="Explicit canvas length; required for a base checkpoint")
    parser.add_argument("--max-len", type=int, help="Upper bound for output length")
    parser.add_argument("--mode", choices=["fast", "reasoning"], default="fast")
    parser.add_argument("--dim", type=int, help="Normalized embedding prefix dimension")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = load_pretrained(args.checkpoint, device=args.device, dtype=dtype).eval()
    tokenizer = load_tokenizer(args.tokenizer or args.checkpoint, model)
    result = infer(
        model, tokenizer, args.text, task=args.task, chat=args.chat, truncate=args.truncate,
        steps=args.steps, rounds=args.rounds, canvas_len=args.canvas_len, max_len=args.max_len,
        block_size=args.block_size,
        mode=args.mode, dim=args.dim,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    console_output = output
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        try:
            console_output.encode(encoding)
        except UnicodeEncodeError:
            # Keep stdout valid JSON on terminals that cannot print every Unicode character.
            console_output = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    print(console_output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
