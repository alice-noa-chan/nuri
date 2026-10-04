"""Frequency-only loss on exactly the masked targets of the MLM evaluation (ARCHITECTURE.md §12).

The base sanity gate compares a checkpoint's held-out masked-token loss
with what a model that knows only token frequencies would score. To be a
fair line, that baseline must use the same documents and the same target
positions. This tool repeats scripts/evaluate.py's MLM suite step by step
(the same document sampling, batching, truncation and masking with the
same seed) without loading a model, and scores every target with a fixed
unigram distribution:

    loss = mean over targets of -log p(token),  p = (count + 1) / (total + V)

Two distributions are reported:
- "fit": counted from the --fit-corpus folders (training text): the honest
  line, since the evaluated text was not seen;
- "eval_fitted": counted from the evaluated corpus itself: optimistic,
  because it knows the held-out text's own frequencies.
A checkpoint has learned context only if its MLM loss on the same
documents is clearly below these lines.

Usage
  python tools/frequency_baseline.py --corpus <held-out folder> --config runs/x/final/config.json \
      --documents 64 --seed 0 --fit-corpus <train folder> [<train folder> ...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nuri.config import PRESETS, NuriConfig  # noqa: E402
from nuri.corruption import mask_for_mlm, pad_batch  # noqa: E402
from nuri.data import TokenCorpus  # noqa: E402
from nuri.evaluation import _batches  # noqa: E402

CHUNK = 50_000_000  # tokens counted at a time, so a multi-billion-token corpus fits in memory


def token_counts(folders: list[str | Path], vocab_size: int) -> np.ndarray:
    """Token frequencies over whole corpus folders (tokens.bin), counted in chunks."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for folder in folders:
        tokens = TokenCorpus(folder).tokens
        for start in range(0, len(tokens), CHUNK):
            counts += np.bincount(np.asarray(tokens[start:start + CHUNK], dtype=np.int64), minlength=vocab_size)
    return counts


def evaluation_targets(corpus: TokenCorpus, config: NuriConfig, documents: int, seed: int,
                       rate: float = 0.15, batch_size: int = 32) -> torch.Tensor:
    """The target token ids of evaluate.py's MLM suite, in the same order.

    Mirrors scripts/evaluate.py (random.Random(seed), corpus.sample with
    len(d) >= 2) and evaluation.mlm_accuracy (batches of 32, first N tokens,
    mask_for_mlm with torch.Generator().manual_seed(seed)).
    """
    docs = corpus.sample(documents, random.Random(seed), usable=lambda d: len(d) >= 2)
    generator = torch.Generator().manual_seed(seed)
    special = (config.pad_id, config.mask_id, config.eos_id)
    labels = []
    for chunk in _batches(list(docs), batch_size):
        ids, valid = pad_batch([d[:config.max_input_len] for d in chunk], config.pad_id)
        _, mask = mask_for_mlm(ids, valid, config.mask_id, config.vocab_size, special, rate, generator)
        labels.append(ids[mask])
    return torch.cat(labels)


def frequency_loss(targets: torch.Tensor, counts: np.ndarray) -> float:
    """Mean -log p(token) under add-one smoothed frequencies."""
    vocab = len(counts)
    log_p = np.log(counts + 1.0) - math.log(counts.sum() + vocab)
    return float(-log_p[targets.numpy()].mean())


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="the held-out corpus folder given to evaluate.py")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="config.json of the checkpoint being evaluated")
    source.add_argument("--preset", choices=sorted(PRESETS), help="or a release preset")
    p.add_argument("--documents", type=int, default=500, help="the same --documents as evaluate.py")
    p.add_argument("--seed", type=int, default=0, help="the same --seed as evaluate.py")
    p.add_argument("--fit-corpus", nargs="*", default=[], help="training corpus folders for the honest line")
    p.add_argument("--output", help="also write the result as JSON here")
    args = p.parse_args(argv)

    config = NuriConfig.load(args.config) if args.config else PRESETS[args.preset]()
    corpus = TokenCorpus(args.corpus)
    targets = evaluation_targets(corpus, config, args.documents, args.seed)
    result = {"documents": args.documents, "seed": args.seed, "targets": int(len(targets)),
              "uniform": math.log(config.vocab_size),
              "eval_fitted": frequency_loss(targets, token_counts([args.corpus], config.vocab_size))}
    if args.fit_corpus:
        result["fit"] = frequency_loss(targets, token_counts(args.fit_corpus, config.vocab_size))
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
