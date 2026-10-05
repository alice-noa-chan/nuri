"""Input-free, per-position frequency line for span reconstruction (ARCHITECTURE.md §12, D-019).

With learned canvas positions (D-019), a readout could lower full-mask span
loss by learning which tokens are common at each output position, without
reading the input at all. To read a corrected model's full-mask result,
compare it with a model that knows only those position-wise frequencies.

The tool draws training spans with the real base-stage splitter
(nuri.corruption.split_span, the same min_span and limits as span
denoising), counts the target token at every canvas position, and scores
given target rows with two input-free distributions:

- "pooled": one unigram over all span-target tokens, add-one smoothed;
- "positional": per position i, smoothed toward the pooled unigram,
      p_i(t) = (count_i(t) + mu * p_pooled(t)) / (n_i + mu)

Both losses are reported as the gate does it (mean over rows of each row's
mean CE) and as a plain token mean, together with how many targets the
argmax token of each distribution gets right. Only the counting uses data;
no model is loaded. The fit corpus must be training text only.

Usage
  python tools/position_frequency_baseline.py --fit-corpus <train folder> [...] \
      --config runs/x/final/config.json --targets targets.json [--samples 200000] [--output line.json]

targets.json holds a list of rows; each row is a list of target token ids in
canvas order (position 0 first), or an object with a "tokens" list.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.config import PRESETS, NuriConfig  # noqa: E402
from nuri.corruption import split_span  # noqa: E402
from nuri.data import open_corpus  # noqa: E402

MIN_SPAN = 16  # the base-stage span denoising minimum (nuri.data.build_span_batch)


def span_position_counts(corpus, config: NuriConfig, samples: int, seed: int) -> np.ndarray:
    """Counts [M, V] of the target token at each canvas position over `samples` training spans."""
    rng = random.Random(seed)
    counts = np.zeros(config.max_response_len * config.vocab_size, dtype=np.int64)
    flat = []
    for doc in corpus.sample(samples, rng, usable=lambda d: len(d) >= 2):
        _, target = split_span(doc, config.max_input_len, config.max_response_len, config.eos_id, MIN_SPAN, rng)
        flat.extend(i * config.vocab_size + t for i, t in enumerate(target))
        if len(flat) > 5_000_000:  # bound memory on large sample counts
            counts += np.bincount(np.asarray(flat, dtype=np.int64), minlength=counts.size)
            flat = []
    if flat:
        counts += np.bincount(np.asarray(flat, dtype=np.int64), minlength=counts.size)
    return counts.reshape(config.max_response_len, config.vocab_size)


def log_probabilities(counts: np.ndarray, prior_strength: float) -> tuple[np.ndarray, np.ndarray]:
    """(pooled log p [V], positional log p [M, V]) from position counts."""
    pooled_counts = counts.sum(axis=0)
    pooled = (pooled_counts + 1.0) / (pooled_counts.sum() + counts.shape[1])
    positional = (counts + prior_strength * pooled) / (counts.sum(axis=1, keepdims=True) + prior_strength)
    return np.log(pooled), np.log(positional)


def score(rows: list[list[int]], pooled: np.ndarray, positional: np.ndarray) -> dict:
    """Row-mean CE, token-mean CE and argmax correct counts of both distributions on target rows."""
    if not rows or any(not row for row in rows):
        raise ValueError("targets need at least one non-empty row")
    if max(len(row) for row in rows) > positional.shape[0]:
        raise ValueError("a target row is longer than max_response_len")
    pooled_best = int(pooled.argmax())
    positional_best = positional.argmax(axis=1)
    result = {"rows": len(rows), "targets": sum(len(row) for row in rows)}
    for name in ("pooled", "positional"):
        row_losses, token_losses, correct = [], [], 0
        for row in rows:
            positions = np.arange(len(row))
            tokens = np.asarray(row)
            log_p = pooled[tokens] if name == "pooled" else positional[positions, tokens]
            best = pooled_best if name == "pooled" else positional_best[positions]
            row_losses.append(float(-log_p.mean()))
            token_losses.extend((-log_p).tolist())
            correct += int((tokens == best).sum())
        result[name] = {"row_mean_ce": float(np.mean(row_losses)), "token_mean_ce": float(np.mean(token_losses)),
                        "correct": correct}
    return result


def read_targets(path: str | Path) -> list[list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [list(map(int, row["tokens"] if isinstance(row, dict) else row)) for row in data]


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-corpus", nargs="+", required=True, help="training corpus folders (never the evaluated text)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="config.json of the checkpoint being evaluated")
    source.add_argument("--preset", choices=sorted(PRESETS), help="or a release preset")
    p.add_argument("--targets", required=True, help="JSON list of target rows in canvas order")
    p.add_argument("--samples", type=int, default=200_000, help="training spans to count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prior-strength", type=float, default=1000.0, help="mu: pull of each position toward pooled")
    p.add_argument("--output", help="also write the result as JSON here")
    args = p.parse_args(argv)

    config = NuriConfig.load(args.config) if args.config else PRESETS[args.preset]()
    counts = span_position_counts(open_corpus(args.fit_corpus), config, args.samples, args.seed)
    pooled, positional = log_probabilities(counts, args.prior_strength)
    result = {"samples": args.samples, "seed": args.seed, "prior_strength": args.prior_strength,
              "uniform": math.log(config.vocab_size), **score(read_targets(args.targets), pooled, positional)}
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
