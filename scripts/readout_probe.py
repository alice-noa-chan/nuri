"""CPU A/B test of readout repairs on real text, starting from a trained checkpoint (Q-17).

Usage
  python scripts/readout_probe.py --init <model folder> --train <corpus> [<corpus>=<weight> ...] \
      --valid <corpus> [...] --out runs/readout_probe

The synthetic diagnostics (docs/GENERATION_DIAGNOSTICS.md) showed that the
readout cannot read the input in order without a position signal, and that
EOS is rarely trained. This script checks whether the package fixes help a
real trained model on real text, without a GPU:

- The encoder, reasoning core and tied token embedding stay frozen. Their
  outputs (X and R, at depth T=4) are computed once per example and cached,
  so each training step only runs the readout.
- Only the readout is trained, for the same number of steps in every arm:
    baseline:    the unchanged readout
    memory:      readout memory positions added (add_memory_positions)
    memory-eos:  memory positions, and half of the training spans end at
                 the document end with EOS (span_end_prob=0.5)
  The memory tables start at zero, so every arm starts from the same function.
- Inputs are cut to --input-len tokens and spans to --span-len tokens to fit
  CPU time. These are short warm-start probes, not release training.

Evaluation uses validation corpora only (never test), the oracle canvas length
and greedy decoding with S=1 and S=4:
- mid: spans cut at random points (no EOS): full-mask cross-entropy, token
  accuracy, token F1 and adjacent-repeat rate.
- end: spans that end at the document end, so the last canvas slot must be
  EOS: the same metrics plus correct EOS placement.
The untrained checkpoint is reported as "initial". Results go to
<out>/results.json; corpus paths are not written to the results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.checkpoint import add_memory_positions, load_pretrained  # noqa: E402
from nuri.corruption import corrupt_canvas, pad_batch, split_span  # noqa: E402
from nuri.data import open_corpus  # noqa: E402
from nuri.evaluation import adjacent_repeat_rate, token_f1  # noqa: E402
from nuri.generation import generate  # noqa: E402
from nuri.losses import vocab_sample_mean_ce  # noqa: E402
from nuri.model import NuriModel  # noqa: E402

ARMS = {
    "baseline": {"memory_positions": False, "span_end_prob": 0.0},
    "memory": {"memory_positions": True, "span_end_prob": 0.0},
    "memory-eos": {"memory_positions": True, "span_end_prob": 0.5},
}
DEPTH = 4
MIN_SPAN = 16  # the base-stage span minimum
FULL_MASK_PROB = 0.1  # the base-stage full-canvas share
WARMUP_STEPS = 50


def make_pairs(
    corpus, count: int, rng: random.Random, args, end_prob: float, eos_id: int
) -> list:
    """(prefix, target) pairs; the target includes EOS only when the span ends the document."""
    documents = corpus.sample(count, rng, usable=lambda d: len(d) > MIN_SPAN + 8)
    return [
        split_span(
            doc, args.input_len, args.span_len + 1, eos_id, MIN_SPAN, rng, end_prob
        )
        for doc in documents
    ]


@torch.no_grad()
def encode_pairs(model: NuriModel, pairs: list, batch_size: int = 16) -> list:
    """Cache (X, R) per example in bfloat16, without padding."""
    cached = []
    for start in range(0, len(pairs), batch_size):
        prefixes = [prefix for prefix, _ in pairs[start : start + batch_size]]
        ids, valid = pad_batch(prefixes, model.config.pad_id)
        x, valid = model.encode(ids, valid)
        r = model.reason(x, valid, DEPTH)
        for row, prefix in enumerate(prefixes):
            cached.append((x[row, : len(prefix)].bfloat16(), r[row].bfloat16()))
    return cached


def memory_batch(model: NuriModel, cached: list, rows: list[int]):
    """Readout memory for some cached examples, padded on the right."""
    width = max(cached[i][0].shape[0] for i in rows)
    hidden = model.config.hidden_size
    x = torch.zeros(len(rows), width, hidden)
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    for slot, i in enumerate(rows):
        length = cached[i][0].shape[0]
        x[slot, :length] = cached[i][0].float()
        valid[slot, :length] = True
    r = torch.stack([cached[i][1].float() for i in rows])
    return model.generation_memory(x, valid, r)


def canvas_cross_entropy(model, cached, targets, rows, canvas, loss_mask):
    """Sample-mean canvas CE for the given rows (same loss as base training)."""
    cache = memory_batch(model, cached, rows)
    target_ids, canvas_valid = pad_batch(
        [targets[i] for i in rows], model.config.pad_id
    )
    hidden = model.readout_hidden(canvas, canvas_valid, cache)
    return vocab_sample_mean_ce(
        hidden[loss_mask], model.embedding.weight, target_ids, loss_mask
    )


def learning_rate_factor(step: int, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_readout(model: NuriModel, pairs: list, cached: list, args, seed: int) -> dict:
    """Train only readout parameters on cached memories. Returns loss summaries."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.readout.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, args.steps)
    )
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    targets = [target for _, target in pairs]
    model.train()
    losses = []
    for step in range(args.steps):
        rows = rng.sample(range(len(pairs)), args.batch_size)
        target_ids, canvas_valid = pad_batch(
            [targets[i] for i in rows], model.config.pad_id
        )
        canvas, loss_mask = corrupt_canvas(
            target_ids,
            canvas_valid,
            model.config.mask_id,
            FULL_MASK_PROB,
            generator=generator,
        )
        loss = canvas_cross_entropy(model, cached, targets, rows, canvas, loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss became {loss.item()} at step {step}")
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            recent = statistics.fmean(losses[-50:])
            print(f"  step {step + 1}/{args.steps}: loss {recent:.4f}", flush=True)
    tail = max(1, args.steps // 10)
    return {
        "first_loss": statistics.fmean(losses[:tail]),
        "final_loss": statistics.fmean(losses[-tail:]),
    }


@torch.no_grad()
def full_mask_cross_entropy(model: NuriModel, pairs: list, cached: list) -> float:
    """Mean sample CE when the whole canvas is MASK (the first fill of generation)."""
    model.eval()
    targets = [target for _, target in pairs]
    values = []
    for start in range(0, len(pairs), 16):
        rows = list(range(start, min(start + 16, len(pairs))))
        target_ids, canvas_valid = pad_batch(
            [targets[i] for i in rows], model.config.pad_id
        )
        canvas = torch.where(canvas_valid, model.config.mask_id, model.config.pad_id)
        loss = canvas_cross_entropy(model, cached, targets, rows, canvas, canvas_valid)
        values.append((loss.item(), len(rows)))
    return sum(v * n for v, n in values) / sum(n for _, n in values)


@torch.no_grad()
def generation_scores(model: NuriModel, pairs: list, rounds: int) -> dict:
    """Greedy decoding at the oracle canvas length.

    eos_correct is measured only over targets that end with EOS (all of the
    "end" set and the few "mid" spans that happen to reach a document end);
    eos_examples says how many there were.
    """
    model.eval()
    eos = model.config.eos_id
    totals = {"token_accuracy": 0.0, "token_f1": 0.0, "repeat_rate": 0.0}
    eos_hits = eos_examples = 0
    for start in range(0, len(pairs), 16):
        chunk = pairs[start : start + 16]
        ids, valid = pad_batch([prefix for prefix, _ in chunk], model.config.pad_id)
        lengths = torch.tensor([len(target) for _, target in chunk])
        canvas = generate(
            model, ids, valid, steps=DEPTH, rounds=rounds, canvas_len=lengths
        ).canvas
        for row, (_, target) in enumerate(chunk):
            predicted = canvas[row, : len(target)].tolist()
            content = [t for t in target if t != eos]
            answer = (
                predicted[: predicted.index(eos)] if eos in predicted else predicted
            )
            totals["token_accuracy"] += sum(
                p == t for p, t in zip(predicted, target)
            ) / len(target)
            totals["token_f1"] += token_f1(answer, content)
            totals["repeat_rate"] += adjacent_repeat_rate(answer)
            if target[-1] == eos:
                eos_examples += 1
                eos_hits += predicted[-1] == eos and eos not in predicted[:-1]
    scores = {name: value / len(pairs) for name, value in totals.items()}
    if eos_examples:
        scores["eos_correct"] = eos_hits / eos_examples
        scores["eos_examples"] = eos_examples
    return scores


def evaluate(model: NuriModel, eval_sets: dict) -> dict:
    report = {}
    for name, (pairs, cached) in eval_sets.items():
        report[name] = {"full_mask_ce": full_mask_cross_entropy(model, pairs, cached)}
        for rounds in (1, 4):
            report[name][f"s{rounds}"] = generation_scores(model, pairs, rounds)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", type=Path, required=True, help="trained model folder")
    parser.add_argument(
        "--train", nargs="+", required=True, help="training corpus folder(s)"
    )
    parser.add_argument(
        "--valid", nargs="+", required=True, help="validation corpus folder(s)"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/readout_probe"))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--pool", type=int, default=2400, help="cached training examples per pool"
    )
    parser.add_argument("--eval-examples", type=int, default=192)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--span-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)
    for name in (
        "steps",
        "batch_size",
        "pool",
        "eval_examples",
        "input_len",
        "span_len",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size > args.pool:
        parser.error("--batch-size cannot exceed --pool")
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = load_pretrained(args.init).eval()
    eos_id = model.config.eos_id
    if (
        args.input_len > model.config.max_input_len
        or args.span_len + 1 > model.config.max_response_len
    ):
        raise ValueError("--input-len/--span-len exceed the model limits")
    train_corpus, valid_corpus = open_corpus(args.train), open_corpus(args.valid)

    print("caching validation memories", flush=True)
    eval_rng = random.Random(args.seed + 1)
    eval_sets = {}
    for name, end_prob in (("mid", 0.0), ("end", 1.0)):
        pairs = make_pairs(
            valid_corpus, args.eval_examples, eval_rng, args, end_prob, eos_id
        )
        eval_sets[name] = (pairs, encode_pairs(model, pairs))

    results = {
        "settings": {
            k: v
            for k, v in vars(args).items()
            if k not in ("init", "train", "valid", "out")
        },
        "depth": DEPTH,
        "arms": {"initial": {"evaluation": evaluate(model, eval_sets)}},
    }
    print(json.dumps(results["arms"]["initial"]), flush=True)

    pools = {}
    for arm in args.arms:
        end_prob = ARMS[arm]["span_end_prob"]
        if end_prob not in pools:
            print(f"caching a training pool (span_end_prob={end_prob})", flush=True)
            pairs = make_pairs(
                train_corpus,
                args.pool,
                random.Random(args.seed),
                args,
                end_prob,
                eos_id,
            )
            pools[end_prob] = (pairs, encode_pairs(model, pairs))
        pairs, cached = pools[end_prob]
        print(f"training arm {arm}", flush=True)
        arm_model = load_pretrained(args.init)
        if ARMS[arm]["memory_positions"]:
            add_memory_positions(arm_model)
        arm_started = time.perf_counter()
        training = train_readout(arm_model, pairs, cached, args, args.seed)
        results["arms"][arm] = {
            "training": training,
            "seconds": round(time.perf_counter() - arm_started, 1),
            "evaluation": evaluate(arm_model, eval_sets),
        }
        print(json.dumps(results["arms"][arm]), flush=True)
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    results["seconds"] = round(time.perf_counter() - started, 1)
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    main()
