"""Training data: an on-disk token corpus and the batch builders for each objective.

Corpus format (a folder):
- tokens.bin   all token ids of all documents, back to back, as uint32
- offsets.npy  int64 array of length num_docs + 1; document i is
               tokens[offsets[i]:offsets[i + 1]]

Several corpus folders can be mixed with weights (MixedCorpus), for example
one folder per language with the planned language ratios.

Dataset locations are always passed in from outside (CLI flags or
environment variables); this module never names a dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .config import NuriConfig
from .corruption import corrupt_canvas, mask_for_mlm, pad_batch, split_span

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


class _DocumentSampler:
    """Shared sampling logic; subclasses provide random_document(rng)."""

    def random_document(self, rng: random.Random) -> list[int]:
        raise NotImplementedError

    def sample(self, count: int, rng: random.Random, usable=None, max_draws_per_doc: int = 100) -> list[list[int]]:
        """Draw `count` random documents (with replacement).

        usable: optional test `document -> bool`. Documents that fail it are
        skipped and redrawn. If too few usable documents turn up, a clear
        error is raised instead of looping forever.
        """
        picked = []
        for _ in range(count * max_draws_per_doc):
            doc = self.random_document(rng)
            if usable is None or usable(doc):
                picked.append(doc)
                if len(picked) == count:
                    return picked
        raise ValueError("the corpus has too few usable documents for this batch")


class TokenCorpus(_DocumentSampler):
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

    def random_document(self, rng: random.Random) -> list[int]:
        return self.document(rng.randrange(len(self)))


class MixedCorpus(_DocumentSampler):
    """Several corpora sampled by weight: each document first picks a source.

    Weights are relative (45/35/10 works as well as 0.45/0.35/0.10). They set
    the share of sampled documents, not of tokens, so sources with longer
    documents contribute more tokens per document.
    """

    def __init__(self, sources: list[tuple[TokenCorpus, float]]):
        if not sources or any(weight <= 0 for _, weight in sources):
            raise ValueError("MixedCorpus needs at least one source and positive weights")
        self.corpora = [corpus for corpus, _ in sources]
        self.weights = [weight for _, weight in sources]

    def __len__(self) -> int:
        return sum(len(corpus) for corpus in self.corpora)

    def random_document(self, rng: random.Random) -> list[int]:
        return rng.choices(self.corpora, self.weights)[0].random_document(rng)


def open_corpus(specs: str | list[str]) -> TokenCorpus | MixedCorpus:
    """Open one corpus folder, or mix several given as "folder" or "folder=weight".

    A single folder without a weight returns a plain TokenCorpus, so existing
    runs keep exactly the same sampling. Missing weights count as 1.
    """
    specs = [specs] if isinstance(specs, str) else list(specs)
    if len(specs) == 1 and "=" not in specs[0]:
        return TokenCorpus(specs[0])
    sources = []
    for spec in specs:
        folder, _, weight = spec.rpartition("=") if "=" in spec else (spec, "", "1")
        sources.append((TokenCorpus(folder), float(weight)))
    return MixedCorpus(sources)


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
    special = (config.pad_id, config.mask_id, config.eos_id)
    windows = [_mlm_window(doc, config.max_input_len, special, rng) for doc in documents]
    ids, valid = pad_batch(windows, config.pad_id)
    corrupted, targets = mask_for_mlm(ids, valid, config.mask_id, config.vocab_size, special, rate, generator)
    # A window made only of special tokens has nothing to predict; drop it.
    keep = targets.any(dim=1)
    return {"input_ids": corrupted[keep], "input_valid": valid[keep], "labels": ids[keep],
            "target_mask": targets[keep]}


def _mlm_window(doc: list[int], length: int, special: tuple[int, ...], rng: random.Random) -> list[int]:
    """At most `length` tokens of doc, anchored on a random maskable token so there is a target."""
    maskable = [i for i, t in enumerate(doc) if t not in special]
    anchor = rng.choice(maskable) if maskable else 0
    lowest = max(0, anchor - length + 1)
    highest = max(lowest, min(anchor, len(doc) - length))
    start = rng.randint(lowest, highest)
    return doc[start:start + length]


def build_packed_mlm_batch(draw_document, rows: int, config: NuriConfig, rng: random.Random,
                           generator: torch.Generator | None = None, rate: float = 0.3,
                           min_piece: int = 16) -> dict:
    """MLM rows filled with several documents, so short documents waste no padding.

    draw_document() returns one usable document. Each row takes document
    windows until less than min_piece tokens of room are left. "segments"
    numbers the documents of a row (1, 2, ...; 0 = padding); the encoder uses
    it to stop attention between documents.
    """
    special = (config.pad_id, config.mask_id, config.eos_id)
    sequences, segment_rows = [], []
    for _ in range(rows):
        tokens, segments = [], []
        while config.max_input_len - len(tokens) >= min(min_piece, config.max_input_len):
            piece = _mlm_window(draw_document(), config.max_input_len - len(tokens), special, rng)
            segments += [segments[-1] + 1 if segments else 1] * len(piece)
            tokens += piece
        sequences.append(tokens)
        segment_rows.append(segments)
    ids, valid = pad_batch(sequences, config.pad_id)
    segment_ids, _ = pad_batch(segment_rows, 0)
    corrupted, targets = mask_for_mlm(ids, valid, config.mask_id, config.vocab_size, special, rate, generator)
    return {"input_ids": corrupted, "input_valid": valid, "labels": ids, "target_mask": targets,
            "segments": segment_ids}


def has_mlm_target(document: list[int], config: NuriConfig) -> bool:
    """True if the document has at least one token that MLM may mask."""
    return any(t not in (config.pad_id, config.mask_id, config.eos_id) for t in document)


def usable_instruction(example: tuple[list[int], list[int]], config: NuriConfig) -> bool:
    """Instruction examples need a prompt and an answer that fits M with EOS."""
    prompt, answer = example
    return bool(prompt) and len(answer) + 1 <= config.max_response_len


def length_window(items: list, sizes: list[tuple[int, int]], input_budget: int, canvas_budget: int,
                  max_rows: int, rng: random.Random) -> list:
    """A run of similar-length items whose padded batch stays within every limit.

    Items are sorted by total length; from a random start, items are taken in
    order while rows x longest input part <= input_budget, rows x longest
    canvas part <= canvas_budget and rows <= max_rows. The input and canvas
    limits are separate, so a run of very short inputs cannot turn into far
    more canvas positions (and vocabulary logits) than a full batch has.
    Similar lengths mean little padding. The function keeps no state between
    calls, so a resumed run draws exactly the same batches.
    """
    order = sorted(range(len(items)), key=lambda i: sum(sizes[i]))
    chosen, longest_a, longest_b = [], 0, 0
    for i in order[rng.randrange(len(order)):]:
        a, b = max(longest_a, sizes[i][0]), max(longest_b, sizes[i][1])
        rows = len(chosen) + 1
        if chosen and (rows * a > input_budget or rows * b > canvas_budget or rows > max_rows):
            break
        chosen.append(i)
        longest_a, longest_b = a, b
    return [items[i] for i in chosen]


def build_span_batch(documents: list[list[int]], config: NuriConfig, rng: random.Random,
                     generator: torch.Generator | None = None, min_span: int = 16,
                     bucket: tuple[int, int, int] | None = None) -> dict:
    """Text span denoising: encode a prefix, rebuild the following span on the canvas.

    bucket=(input budget, canvas budget, max rows): pass more documents than one batch needs;
    a run of similar-length samples is kept (see length_window).
    """
    pairs = [split_span(doc, config.max_input_len, config.max_response_len, config.eos_id, min_span, rng)
             for doc in documents if len(doc) >= 2]
    if bucket is not None:
        pairs = length_window(pairs, [(len(s), len(t)) for s, t in pairs], *bucket, rng)
    return _canvas_batch([p[0] for p in pairs], [p[1] for p in pairs], config, generator)


def build_instruction_batch(examples: list[tuple[list[int], list[int]]], config: NuriConfig,
                            generator: torch.Generator | None = None,
                            bucket: tuple[int, int, int] | None = None, rng: random.Random | None = None) -> dict:
    """examples: (prompt ids, answer ids without EOS).

    EOS is appended to every answer (a real ending). Answers longer than M
    after EOS are dropped, never cut, and counted in "dropped". Prompts keep
    their last N tokens, since the end of a conversation matters most.
    bucket=(input budget, canvas budget, max rows) with rng keeps only a run of
    similar-length examples (see length_window).
    """
    if bucket is not None:
        usable = [e for e in examples if usable_instruction(e, config)]
        sizes = [(min(len(p), config.max_input_len), len(a) + 1) for p, a in usable]
        examples = length_window(usable, sizes, *bucket, rng) if usable else []
    prompts, targets = [], []
    for prompt, answer in examples:
        if not usable_instruction((prompt, answer), config):
            continue
        target = list(answer) + [config.eos_id]
        prompts.append(list(prompt)[-config.max_input_len:])
        targets.append(target)
    batch = _canvas_batch(prompts, targets, config, generator) if prompts else {}
    if prompts:
        batch["lengths"] = torch.tensor([len(t) for t in targets])
    batch["dropped"] = len(examples) - len(prompts)
    return batch


def build_pair_batch(pairs: list[tuple[list[int], list[int]]], config: NuriConfig) -> dict:
    """Query/document pairs for contrastive training; row i of each side is a positive pair.

    "false_negatives"[i, j] marks rows j != i that must not count as negatives
    for row i: the same document text, or the same query text (whose
    document is then also a valid answer for query i).
    """
    queries = [list(q[:config.max_input_len]) for q, _ in pairs]
    documents = [list(d[:config.max_input_len]) for _, d in pairs]
    n = len(pairs)
    false_negatives = torch.tensor([[i != j and (documents[i] == documents[j] or queries[i] == queries[j])
                                     for j in range(n)] for i in range(n)], dtype=torch.bool)
    query_ids, q_valid = pad_batch(queries, config.pad_id)
    doc_ids, d_valid = pad_batch(documents, config.pad_id)
    return {"query_ids": query_ids, "query_valid": q_valid, "doc_ids": doc_ids, "doc_valid": d_valid,
            "false_negatives": false_negatives}


def _canvas_batch(sources, targets, config: NuriConfig, generator) -> dict:
    input_ids, input_valid = pad_batch(sources, config.pad_id)
    target_ids, canvas_valid = pad_batch(targets, config.pad_id)
    canvas, loss_mask = corrupt_canvas(target_ids, canvas_valid, config.mask_id, generator=generator)
    return {"input_ids": input_ids, "input_valid": input_valid, "canvas": canvas,
            "canvas_valid": canvas_valid, "targets": target_ids, "loss_mask": loss_mask}
