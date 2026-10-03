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
from .data import (TokenCorpus, build_instruction_batch, build_mlm_batch, build_pair_batch, build_span_batch,
                       has_mlm_target, usable_instruction)
from .trainer import Task


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _corpus_tasks(corpus: TokenCorpus, config: NuriConfig, batch_size: int, seed: int) -> dict[str, Task]:
    """MLM and span batches drawn from the corpus, skipping documents they cannot use."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    def mlm_batch() -> dict:
        docs = corpus.sample(batch_size, rng, usable=lambda d: has_mlm_target(d, config))
        return build_mlm_batch(docs, config, rng, generator)

    def span_batch() -> dict:
        docs = corpus.sample(batch_size, rng, usable=lambda d: len(d) >= 2)  # one token on each side
        return build_span_batch(docs, config, rng, generator)

    return {"mlm": Task("mlm", mlm_batch, rng, generator), "span": Task("span", span_batch, rng, generator)}


def _usable_examples(examples: list, config: NuriConfig, name: str) -> list:
    """Filter the pool once up front, so batch building never has to retry."""
    usable = [e for e in examples if usable_instruction(e, config)]
    if not usable:
        raise ValueError(f"no usable {name} examples (each needs a prompt and an answer of at most M-1 tokens)")
    return usable


def _example_task(kind: str, examples: list, config: NuriConfig, batch_size: int, seed: int) -> Task:
    """Endless random batches over a pre-filtered list of (prompt ids, answer ids)."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    return Task(kind, lambda: build_instruction_batch(rng.sample(examples, min(batch_size, len(examples))),
                                                      config, generator), rng, generator)


def base_stage(corpus: TokenCorpus, config: NuriConfig, batch_size: int, seed: int = 0):
    return _corpus_tasks(corpus, config, batch_size, seed), {"mlm": 0.5, "span": 0.5}


def instruction_stage(examples: list, config: NuriConfig, batch_size: int, corpus: TokenCorpus | None = None,
                      reasoning: list | None = None, seed: int = 0):
    examples = _usable_examples(examples, config, "instruction")
    tasks = {"instruction": _example_task("instruction", examples, config, batch_size, seed)}
    mix = {"instruction": 0.7}
    if reasoning:
        reasoning = _usable_examples(reasoning, config, "reasoning")
        tasks["reasoning"] = _example_task("instruction", reasoning, config, batch_size, seed + 1)
        mix["reasoning"] = 0.15
    else:
        mix["instruction"] += 0.15
    if corpus is not None:
        tasks["span"] = _corpus_tasks(corpus, config, batch_size, seed + 2)["span"]
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
    task = Task("pairs", lambda: build_pair_batch(rng.sample(pairs, min(batch_size, len(pairs))), config), rng)
    return {"pairs": task}, {"pairs": 1.0}
