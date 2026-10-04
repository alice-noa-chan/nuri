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
  retrieval      Recall@k / MRR on --pairs (JSONL with "query" and "positive")
  diagnostics    per-step core state statistics
  instructions   generation metrics and coverage on --instructions JSONL ("messages", "response");
                 oracle length, plus predicted length for a trained length head

Base models have no trained length head, so their generation suites use the
oracle length only, and the report says so.
"""

from __future__ import annotations

import argparse
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
from nuri.tasks import reasoning_examples  # noqa: E402
from nuri.tokenizer import check_tokenizer, load_tokenizer  # noqa: E402

SUITES = ("reasoning", "curves", "interventions", "halting", "mlm", "denoise", "retrieval", "diagnostics",
          "instructions")


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


def main(argv=None) -> dict:
    args = parse_args(argv)
    if "instructions" in args.suites and not args.instructions:
        sys.exit("instructions needs --instructions (or NURI_EVAL_INSTRUCTIONS)")
    model = load_pretrained(args.checkpoint, device=args.device).eval()
    tokenizer = load_tokenizer(args.tokenizer or args.checkpoint)
    check_tokenizer(tokenizer, model.config)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
    natural_examples = instruction_examples(args.instructions, encode) if "instructions" in args.suites else None

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
        if not args.pairs:
            sys.exit("retrieval needs --pairs (or NURI_EVAL_PAIRS)")
        pairs = [(encode(r["query"]), encode(r["positive"])) for r in read_jsonl(args.pairs)]
        pairs = [(q, d) for q, d in pairs if q and d]
        report["retrieval"] = [evaluation.retrieval_metrics(model, pairs, mode=mode, dim=dim)
                               for mode in ("fast", "reasoning") for dim in (64, 128, None)
                               if dim is None or dim <= model.config.embed_dim]
        # Reasoning queries against fast documents; meaningful only after cross-mode alignment (§6).
        report["retrieval"].append(evaluation.retrieval_metrics(model, pairs, mode="reasoning", doc_mode="fast"))

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text.encode("ascii", "backslashreplace").decode("ascii"))  # safe on any console encoding
    return report


if __name__ == "__main__":
    main()
