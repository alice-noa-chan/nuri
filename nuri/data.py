"""Training data: an on-disk token corpus and the batch builders for each objective.

Corpus format (a folder):
- tokens.bin   all token ids of all documents, back to back, as uint32
- offsets.npy  int64 array of length num_docs + 1; document i is
               tokens[offsets[i]:offsets[i + 1]]

Dataset locations are always passed in from outside (CLI flags or
environment variables); this module never names a dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from nuri.config import NuriConfig
from nuri.corruption import corrupt_canvas, mask_for_mlm, pad_batch, split_span

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


class TokenCorpus:
    """Read-only view of a tokenized corpus folder (memory-mapped, so it can be huge)."""

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.tokens = np.memmap(folder / "tokens.bin", dtype=np.uint32, mode="r")
        self.offsets = np.load(folder / "offsets.npy")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def document(self, index: int) -> list[int]:
        start, end = self.offsets[index], self.offsets[index + 1]
        return self.tokens[start:end].astype(np.int64).tolist()

    def sample(self, count: int, rng: random.Random) -> list[list[int]]:
        return [self.document(rng.randrange(len(self))) for _ in range(count)]


def write_corpus(folder: str | Path, documents: Iterable[Sequence[int]]) -> int:
    """Write documents (token id lists) in the corpus format. Returns the number written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    with open(folder / "tokens.bin", "wb") as out:
        for doc in documents:
            if len(doc) == 0:
                continue
            out.write(np.asarray(doc, dtype=np.uint32).tobytes())
            offsets.append(offsets[-1] + len(doc))
    np.save(folder / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    return len(offsets) - 1


def chat_prompt_text(messages: list[dict]) -> str:
    """Render the conversation before the answer, ending with the assistant tag.

    Matches the tokenizer chat template: each turn is <|role|>content[EOS].
    The answer itself goes to the output canvas, never into this prompt.
    """
    turns = "".join(f"{ROLE_TOKENS[m['role']]}{m['content']}[EOS]" for m in messages)
    return turns + ROLE_TOKENS["assistant"]


# ----- batch builders ----------------------------------------------------------------
# Each returns a dict of tensors ready for the training step functions.


def build_mlm_batch(documents: list[list[int]], config: NuriConfig, rng: random.Random,
                    generator: torch.Generator | None = None, rate: float = 0.3) -> dict:
    """Random windows of at most N tokens, masked for MLM."""
    windows = []
    for doc in documents:
        start = rng.randint(0, max(0, len(doc) - config.max_input_len))
        windows.append(doc[start:start + config.max_input_len])
    ids, valid = pad_batch(windows, config.pad_id)
    special = (config.pad_id, config.mask_id, config.eos_id)
    corrupted, targets = mask_for_mlm(ids, valid, config.mask_id, config.vocab_size, special, rate, generator)
    return {"input_ids": corrupted, "input_valid": valid, "labels": ids, "target_mask": targets}


def build_span_batch(documents: list[list[int]], config: NuriConfig, rng: random.Random,
                     generator: torch.Generator | None = None, min_span: int = 16) -> dict:
    """Text span denoising: encode a prefix, rebuild the following span on the canvas."""
    pairs = [split_span(doc, config.max_input_len, config.max_response_len, config.eos_id, min_span, rng)
             for doc in documents if len(doc) >= 2]
    return _canvas_batch([p[0] for p in pairs], [p[1] for p in pairs], config, generator)


def build_instruction_batch(examples: list[tuple[list[int], list[int]]], config: NuriConfig,
                            generator: torch.Generator | None = None) -> dict:
    """examples: (prompt ids, answer ids without EOS).

    EOS is appended to every answer (a real ending). Answers longer than M
    after EOS are dropped, never cut, and counted in "dropped". Prompts keep
    their last N tokens, since the end of a conversation matters most.
    """
    prompts, targets = [], []
    for prompt, answer in examples:
        target = list(answer) + [config.eos_id]
        if len(target) > config.max_response_len or not prompt:
            continue
        prompts.append(list(prompt)[-config.max_input_len:])
        targets.append(target)
    batch = _canvas_batch(prompts, targets, config, generator) if prompts else {}
    if prompts:
        batch["lengths"] = torch.tensor([len(t) for t in targets])
    batch["dropped"] = len(examples) - len(prompts)
    return batch


def build_pair_batch(pairs: list[tuple[list[int], list[int]]], config: NuriConfig) -> dict:
    """Query/document pairs for contrastive training; row i of each side is a positive pair."""
    queries, q_valid = pad_batch([q[:config.max_input_len] for q, _ in pairs], config.pad_id)
    documents, d_valid = pad_batch([d[:config.max_input_len] for _, d in pairs], config.pad_id)
    return {"query_ids": queries, "query_valid": q_valid, "doc_ids": documents, "doc_valid": d_valid}


def _canvas_batch(sources, targets, config: NuriConfig, generator) -> dict:
    input_ids, input_valid = pad_batch(sources, config.pad_id)
    target_ids, canvas_valid = pad_batch(targets, config.pad_id)
    canvas, loss_mask = corrupt_canvas(target_ids, canvas_valid, config.mask_id, generator=generator)
    return {"input_ids": input_ids, "input_valid": input_valid, "canvas": canvas,
            "canvas_valid": canvas_valid, "targets": target_ids, "loss_mask": loss_mask}
