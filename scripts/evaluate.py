"""Run the first-day evaluations of ARCHITECTURE.md §12 on one checkpoint.

Examples
  # reasoning tasks (eval split), T and S curves, R interventions, length check
  python scripts/evaluate.py --checkpoint runs/nuri-base-it/final --suites reasoning curves interventions \
      --output runs/eval/nuri-base-it.json

  # masked-token accuracy on held-out text and core diagnostics
  python scripts/evaluate.py --checkpoint runs/nuri-base/final --suites mlm diagnostics --corpus $NURI_EVAL_CORPUS

  # retrieval with both embedding modes and Matryoshka sizes
  python scripts/evaluate.py --checkpoint runs/nuri-base-embed/final --suites retrieval --pairs $NURI_EVAL_PAIRS

Suites
  reasoning      exact match on synthetic multi-step tasks; predicted and oracle length;
                 on the "shared" split (training pools, new combinations) and the
                 "eval" split (unseen names: a stress test)
  curves         the same metric at T = 1/2/4/8/16 and S = 1/2/4/8 (shared split)
  interventions  normal R vs zeroed R vs R shuffled between samples (shared split)
  halting        exact match vs mean core steps with training-free halting (shared split, up to T=16)
  mlm            masked-token accuracy on --corpus (a held-out corpus folder)
  denoise        held-out span reconstruction loss at 15/50/100% masking, at T = 1/4/8
                 (the core + readout path; works for base models)
  retrieval      Recall@k / MRR and query-overlap strata on --pairs ("query", "positive");
                 each stratum keeps the same complete document candidate pool
  diagnostics    per-step core state statistics
  instructions   generation metrics and coverage on --instructions JSONL ("messages", "response");
                 oracle length, plus predicted length for a trained length head
  conditioning   bounded length-matched prompt replacements on --instructions;
                 correctness with EOS and input sensitivity, reported separately

Base models have no trained length head, so their generation suites use the
oracle length only, and the report says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri import evaluation  # noqa: E402
from nuri.checkpoint import load_pretrained  # noqa: E402
from nuri.data import TokenCorpus, chat_prompt_text  # noqa: E402
from nuri.pipeline import read_jsonl  # noqa: E402
from nuri.retrieval_report import overlap_stratified_retrieval  # noqa: E402
from nuri.tasks import reasoning_examples  # noqa: E402
from nuri.tokenizer import check_tokenizer, load_tokenizer  # noqa: E402

SUITES = ("reasoning", "curves", "interventions", "halting", "mlm", "denoise", "retrieval", "diagnostics",
          "instructions", "conditioning")


def conditioning_limit(value):
    number = int(value)
    if not 1 <= number <= 1024:
        raise argparse.ArgumentTypeError("must be in [1, 1024]")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suites", nargs="+", choices=SUITES, required=True)
    p.add_argument("--tokenizer", help="defaults to the tokenizer inside --checkpoint")
    p.add_argument("--corpus", default=os.environ.get("NURI_EVAL_CORPUS"), help="held-out corpus folder (mlm)")
    p.add_argument("--pairs", default=os.environ.get("NURI_EVAL_PAIRS"), help="held-out pairs JSONL (retrieval)")
    p.add_argument("--instructions", default=os.environ.get("NURI_EVAL_INSTRUCTIONS"),
                   help="held-out canonical instruction JSONL; evaluate the whole file")
    p.add_argument("--examples", type=int, default=300, help="synthetic examples per reasoning generation suite")
    p.add_argument("--documents", type=int, default=500, help="documents for mlm and diagnostics")
    p.add_argument("--conditioning-examples", type=conditioning_limit, default=128,
                   help="conditioning: deterministic maximum selected rows, before length-matched pairing")
    p.add_argument("--conditioning-samples-output",
                   help="optional fresh JSONL under .private for up to eight decoded source-swap pairs per mode")
    p.add_argument("--conditioning-tensorboard-dir", help="optional fresh directory for numeric VALID conditioning curves")
    p.add_argument("--conditioning-language", choices=("ko", "en", "ja"),
                   help="declared language of the reviewed file; required for conditioning TensorBoard")
    p.add_argument("--conditioning-step", type=nonnegative_int, default=0,
                   help="completed optimizer updates of the assessed snapshot (default 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", help="write the report here as JSON")
    return p.parse_args(argv)


def instruction_examples(path: str | Path, encode) -> list[tuple[list[int], list[int]]]:
    """Read the training CLI's canonical rows without filtering or cutting answers."""
    examples = []
    try:
        for index, row in enumerate(read_jsonl(path), start=1):
            if not isinstance(row, dict) or not isinstance(row.get("response"), str):
                sys.exit(f"instructions record {index} needs a string response (empty is allowed)")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                sys.exit(f"instructions record {index} needs a nonempty messages list")
            for message in messages:
                if (not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
                        or not isinstance(message.get("content"), str)):
                    sys.exit(f"instructions record {index} needs valid role/content messages")
            # Prompt role tags and turn EOS match training. Reference EOS is added by the evaluator.
            examples.append((encode(chat_prompt_text(messages)), encode(row["response"])))
    except OSError:
        sys.exit("instructions input is unavailable or unreadable")
    except (json.JSONDecodeError, UnicodeError):
        sys.exit("instructions input must contain valid UTF-8 JSONL")
    return examples


def pair_examples(path: str | Path, encode) -> tuple[list, dict]:
    """Validate the whole supplied file, counting empty token sides explicitly."""
    pairs, supplied = [], 0
    try:
        for index, row in enumerate(read_jsonl(path), start=1):
            if (not isinstance(row, dict) or not isinstance(row.get("query"), str)
                    or not isinstance(row.get("positive"), str)):
                sys.exit(f"pairs record {index} needs string query and positive fields")
            supplied += 1
            query, document = encode(row["query"]), encode(row["positive"])
            if query and document:
                pairs.append((query, document))
    except OSError:
        sys.exit("pairs input is unavailable or unreadable")
    except (json.JSONDecodeError, UnicodeError):
        sys.exit("pairs input must contain valid UTF-8 JSONL")
    return pairs, {"supplied_pairs": supplied, "evaluated_pairs": len(pairs),
                   "excluded_empty_pairs": supplied - len(pairs)}


def main(argv=None) -> dict:
    args = parse_args(argv)
    sample_path = None
    tensorboard_path = None
    if args.conditioning_tensorboard_dir:
        if "conditioning" not in args.suites or not args.conditioning_language:
            sys.exit("conditioning TensorBoard requires the suite and an explicit language")
        tensorboard_path = Path(args.conditioning_tensorboard_dir)
        if tensorboard_path.exists():
            sys.exit("conditioning TensorBoard directory already exists; use a fresh path")
    if args.conditioning_samples_output:
        if "conditioning" not in args.suites:
            sys.exit("--conditioning-samples-output requires the conditioning suite")
        sample_path = Path(args.conditioning_samples_output).resolve()
        if not sample_path.is_relative_to(Path(__file__).resolve().parent.parent / ".private"):
            sys.exit("conditioning samples must stay under the project's .private directory")
        if sample_path.exists():
            sys.exit("conditioning samples output already exists; use a fresh path")
    if "conditioning" in args.suites and args.output and Path(args.output).exists():
        sys.exit("conditioning report already exists; use a fresh path")
    if sample_path is not None and args.output and sample_path == Path(args.output).resolve():
        sys.exit("conditioning samples and aggregate report need different output paths")
    if "instructions" in args.suites and not args.instructions:
        sys.exit("instructions needs --instructions (or NURI_EVAL_INSTRUCTIONS)")
    if "conditioning" in args.suites and not args.instructions:
        sys.exit("conditioning needs --instructions (or NURI_EVAL_INSTRUCTIONS)")
    if "retrieval" in args.suites and not args.pairs:
        sys.exit("retrieval needs --pairs (or NURI_EVAL_PAIRS)")
    model = load_pretrained(args.checkpoint, device=args.device).eval()
    tokenizer = load_tokenizer(args.tokenizer or args.checkpoint)
    check_tokenizer(tokenizer, model.config)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
    natural_examples = instruction_examples(args.instructions, encode) if {"instructions", "conditioning"} & set(args.suites) else None
    retrieval_pairs, retrieval_input = pair_examples(args.pairs, encode) if "retrieval" in args.suites else (None, None)

    report = {"checkpoint": str(args.checkpoint), "length_head_trained": model.config.length_head_trained}
    needs_examples = {"reasoning", "curves", "interventions", "halting"} & set(args.suites)
    if needs_examples:
        # A seed far from the training seeds, so "shared" examples are new combinations.
        def split_examples(split):
            return [(encode(chat_prompt_text(m)), encode(a))
                    for m, a in reasoning_examples(args.examples, seed=1_000_003 + args.seed, split=split)]

        examples = split_examples("shared")
        oracle = not model.config.length_head_trained  # base models cannot predict the length
        report["oracle_length_only"] = oracle
    if "reasoning" in args.suites:
        report["reasoning"] = {}
        for split, split_set in (("shared", examples), ("eval", split_examples("eval"))):
            result = {"oracle_length": evaluation.generation_accuracy(model, split_set, oracle_length=True)}
            if not oracle:
                result["predicted_length"] = evaluation.generation_accuracy(model, split_set)
            report["reasoning"][split] = result
    if "curves" in args.suites:
        report["depth_curve"] = evaluation.depth_curve(model, examples, oracle_length=oracle)
        report["rounds_curve"] = evaluation.rounds_curve(model, examples, oracle_length=oracle)
    if "interventions" in args.suites:
        report["interventions"] = evaluation.r_interventions(model, examples, oracle_length=oracle)
    if "halting" in args.suites:
        report["halting"] = evaluation.halting_curve(model, examples, oracle_length=oracle)

    if "instructions" in args.suites:
        oracle = not model.config.length_head_trained
        report["oracle_length_only"] = oracle
        result = {"input_examples": len(natural_examples), "oracle_length_only": oracle,
                  "oracle_length": evaluation.generation_accuracy(model, natural_examples, oracle_length=True)}
        if not oracle:
            result["predicted_length"] = evaluation.generation_accuracy(model, natural_examples)
        report["instructions"] = result

    if "conditioning" in args.suites:
        controls = {"max_examples": args.conditioning_examples, "seed": args.seed,
                    "sample_count": 8 if sample_path is not None else 0}
        result = {"oracle_length_only": not model.config.length_head_trained,
                  "oracle_length": evaluation.source_conditioning(model, natural_examples, **controls)}
        if model.config.length_head_trained:
            result["predicted_length"] = evaluation.source_conditioning(model, natural_examples,
                                                                         oracle_length=False, **controls)
        if sample_path is not None:
            private_samples = []
            for mode in ("oracle_length", "predicted_length"):
                for sample in result.get(mode, {}).pop("samples", []):
                    decoded = {"mode": mode, **sample}
                    for key in ("original_prompt", "swapped_prompt", "original_reference",
                                "swapped_reference", "original_tokens", "swapped_tokens"):
                        decoded[key + "_text"] = tokenizer.decode(sample[key], skip_special_tokens=False)
                    private_samples.append(decoded)
            payload = "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in private_samples)
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
            result["private_samples"] = {"rows": len(private_samples),
                                         "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
        if tensorboard_path is not None:
            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter(str(tensorboard_path)) as writer:
                for mode in ("oracle_length", "predicted_length"):
                    if mode not in result:
                        continue
                    prefix = f"valid/{args.conditioning_language}/conditioning/{mode}"
                    for key, value in result[mode].items():
                        if type(value) in (int, float):
                            writer.add_scalar(f"{prefix}/{key}", value, args.conditioning_step)
                    writer.add_scalar(f"{prefix}/unassessed", result[mode]["status"] != "assessed",
                                      args.conditioning_step)
            result["tensorboard_written"] = True
        result["declared_language"] = args.conditioning_language
        result["completed_updates"] = args.conditioning_step
        report["conditioning"] = result

    if {"mlm", "denoise", "diagnostics"} & set(args.suites):
        if not args.corpus:
            sys.exit("mlm, denoise and diagnostics need --corpus (or NURI_EVAL_CORPUS)")
        corpus = TokenCorpus(args.corpus)
        rng = random.Random(args.seed)
        documents = corpus.sample(args.documents, rng, usable=lambda d: len(d) >= 2)
        if "mlm" in args.suites:
            report["mlm"] = evaluation.mlm_accuracy(model, documents, seed=args.seed)
        if "denoise" in args.suites:
            report["denoise"] = [row for t in (1, 4, 8)
                                 for row in evaluation.span_denoising(model, documents, steps=t, seed=args.seed)]
        if "diagnostics" in args.suites:
            report["diagnostics"] = evaluation.core_diagnostics(model, documents[:64], steps=16)

    if "retrieval" in args.suites:
        report["retrieval_input"] = retrieval_input
        special_ids = set(token_id for token_id, token in tokenizer.get_added_tokens_decoder().items()
                          if token.special)
        special_ids.update((model.config.pad_id, model.config.mask_id, model.config.eos_id))
        overlap = overlap_stratified_retrieval(model, retrieval_pairs, special_ids=sorted(special_ids))
        report["retrieval_overlap"] = overlap
        # Reuse the overall ranks: strata do not trigger extra model scoring.
        report["retrieval"] = overlap["overall"]["model"]
        report["retrieval_lexical_baseline"] = overlap["overall"]["lexical_tfidf"]

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if "conditioning" in args.suites:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            Path(args.output).write_text(text, encoding="utf-8")
    print(text.encode("ascii", "backslashreplace").decode("ascii"))  # safe on any console encoding
    return report


if __name__ == "__main__":
    main()
