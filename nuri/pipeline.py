"""Glue between data and the training loop: the task mix of each release stage (§8-C).

| stage | tasks (mix weight)                                         |
|-------|------------------------------------------------------------|
| base  | mlm 0.5, span 0.5                                          |
| it    | instruction 0.7, reasoning 0.15, span 0.15 (keeps skills)  |
| embed | pairs 1.0                                                  |

When the IT stage has no reasoning examples, its weight is given to the
instruction task. Everything here takes token ids; tokenizing text is done
by the caller (see scripts/train.py).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch

from .config import NuriConfig
from .data import (TokenCorpus, build_instruction_batch, build_mlm_batch, build_packed_mlm_batch, build_pair_batch,
                       build_span_batch, has_mlm_target, usable_instruction)
from .trainer import Task


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# Length bucketing draws this many times a batch of candidates and keeps a
# similar-length run. The run's padded input and canvas each stay within one
# full batch, and it has at most BUCKET_MAX_ROWS times the nominal rows, which
# bounds the latent-slot work of the core (rows x K).
BUCKET_POOL = 8
BUCKET_MAX_ROWS = 4


def _bucket(config: NuriConfig, batch_size: int) -> tuple[int, int, int]:
    """(input budget, canvas budget, max rows) for one bucketed batch."""
    return (batch_size * config.max_input_len, batch_size * config.max_response_len,
            batch_size * BUCKET_MAX_ROWS)


def _bucket_options(config: NuriConfig, batch_size: int, enabled: bool) -> dict:
    """Persist the selection policy so older/different buckets cannot resume silently."""
    if not enabled:
        return {}
    return {"bucket_policy": {"version": 2, "candidate_multiplier": BUCKET_POOL,
                              "limits": list(_bucket(config, batch_size))}}


def _corpus_tasks(corpus: TokenCorpus, config: NuriConfig, batch_size: int, seed: int,
                  pack_mlm: bool = False, bucket: bool = False) -> dict[str, Task]:
    """MLM and span batches drawn from the corpus, skipping documents they cannot use.

    pack_mlm fills each MLM row with several documents (no padding waste for
    short documents); span batches always hold one document per row. bucket
    groups span samples of similar length, so short samples are not padded to
    the longest one in the batch.
    """
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    mlm_usable = lambda d: has_mlm_target(d, config)  # noqa: E731

    def mlm_batch() -> dict:
        if pack_mlm:
            return build_packed_mlm_batch(lambda: corpus.sample(1, rng, usable=mlm_usable)[0], batch_size,
                                          config, rng, generator)
        docs = corpus.sample(batch_size, rng, usable=mlm_usable)
        return build_mlm_batch(docs, config, rng, generator)

    def span_batch() -> dict:
        count = batch_size * BUCKET_POOL if bucket else batch_size
        docs = corpus.sample(count, rng, usable=lambda d: len(d) >= 2)  # one token on each side
        return build_span_batch(docs, config, rng, generator, bucket=_bucket(config, batch_size) if bucket else None)

    options = {"batch_size": batch_size, "corpus_documents": len(corpus),
               "corpus_weights": getattr(corpus, "weights", None),  # set for a MixedCorpus
               "pack_mlm": pack_mlm, "bucket": bucket,
               **_bucket_options(config, batch_size, bucket)}
    return {"mlm": Task("mlm", mlm_batch, rng, generator, options),
            "span": Task("span", span_batch, rng, generator, options)}


def _usable_examples(examples: list, config: NuriConfig, name: str) -> list:
    """Filter the pool once up front, so batch building never has to retry."""
    usable = [e for e in examples if usable_instruction(e, config)]
    if not usable:
        raise ValueError(f"no usable {name} examples (each needs a prompt and an answer of at most M-1 tokens)")
    return usable


def _example_task(kind: str, examples: list, config: NuriConfig, batch_size: int, seed: int,
                  bucket: bool = False) -> Task:
    """Endless random batches over a pre-filtered list of (prompt ids, answer ids)."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    def next_batch() -> dict:
        if not bucket:
            return build_instruction_batch(rng.sample(examples, min(batch_size, len(examples))), config, generator)
        pool = rng.sample(examples, min(batch_size * BUCKET_POOL, len(examples)))
        return build_instruction_batch(pool, config, generator, bucket=_bucket(config, batch_size), rng=rng)

    return Task(kind, next_batch, rng, generator,
                {"batch_size": batch_size, "examples": len(examples), "bucket": bucket,
                 **_bucket_options(config, batch_size, bucket)})


def base_stage(corpus: TokenCorpus, config: NuriConfig, batch_size: int, seed: int = 0, pack_mlm: bool = False,
               bucket: bool = False):
    return _corpus_tasks(corpus, config, batch_size, seed, pack_mlm, bucket), {"mlm": 0.5, "span": 0.5}


def instruction_stage(examples: list, config: NuriConfig, batch_size: int, corpus: TokenCorpus | None = None,
                      reasoning: list | None = None, seed: int = 0, bucket: bool = False):
    examples = _usable_examples(examples, config, "instruction")
    tasks = {"instruction": _example_task("instruction", examples, config, batch_size, seed, bucket)}
    mix = {"instruction": 0.7}
    if reasoning:
        reasoning = _usable_examples(reasoning, config, "reasoning")
        tasks["reasoning"] = _example_task("instruction", reasoning, config, batch_size, seed + 1, bucket)
        mix["reasoning"] = 0.15
    else:
        mix["instruction"] += 0.15
    if corpus is not None:
        tasks["span"] = _corpus_tasks(corpus, config, batch_size, seed + 2, bucket=bucket)["span"]
        mix["span"] = 0.15
    else:
        mix["instruction"] += 0.15
    return tasks, mix


def embedding_stage(pairs: list, config: NuriConfig, batch_size: int, seed: int = 0):
    # A side that tokenized to nothing cannot be encoded; drop such pairs.
    pairs = [(q, d) for q, d in pairs if q and d]
    # In-batch InfoNCE needs at least one other row as a negative.
    if batch_size < 2 or len(pairs) < 2:
        raise ValueError("embedding training needs batch_size >= 2 and at least 2 non-empty pairs")
    rng = random.Random(seed)
    task = Task("pairs", lambda: build_pair_batch(rng.sample(pairs, min(batch_size, len(pairs))), config), rng,
                options={"batch_size": batch_size, "pairs": len(pairs)})
    return {"pairs": task}, {"pairs": 1.0}
