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

from nuri.config import NuriConfig
from nuri.data import TokenCorpus, build_instruction_batch, build_mlm_batch, build_pair_batch, build_span_batch
from nuri.trainer import Task


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _corpus_tasks(corpus: TokenCorpus, config: NuriConfig, batch_size: int, seed: int) -> dict[str, Task]:
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    return {
        "mlm": Task("mlm", lambda: build_mlm_batch(corpus.sample(batch_size, rng), config, rng, generator)),
        "span": Task("span", lambda: build_span_batch(corpus.sample(batch_size, rng), config, rng, generator)),
    }


def _example_task(kind: str, examples: list, config: NuriConfig, batch_size: int, seed: int) -> Task:
    """Endless shuffled batches over a list of (prompt ids, answer ids) examples."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    def next_batch() -> dict:
        while True:
            batch = build_instruction_batch(rng.sample(examples, min(batch_size, len(examples))), config, generator)
            if "input_ids" in batch:  # skip the rare batch where every answer was too long
                return batch
    return Task(kind, next_batch)


def base_stage(corpus: TokenCorpus, config: NuriConfig, batch_size: int, seed: int = 0):
    return _corpus_tasks(corpus, config, batch_size, seed), {"mlm": 0.5, "span": 0.5}


def instruction_stage(examples: list, config: NuriConfig, batch_size: int, corpus: TokenCorpus | None = None,
                      reasoning: list | None = None, seed: int = 0):
    tasks = {"instruction": _example_task("instruction", examples, config, batch_size, seed)}
    mix = {"instruction": 0.7}
    if reasoning:
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
    rng = random.Random(seed)
    task = Task("pairs", lambda: build_pair_batch(rng.sample(pairs, min(batch_size, len(pairs))), config))
    return {"pairs": task}, {"pairs": 1.0}
