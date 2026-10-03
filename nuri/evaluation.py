"""Evaluations from ARCHITECTURE.md §12.

All functions take token ids, so they work with any tokenizer, and return
plain dicts that can be written as JSON. Every result reports its settings
(T, S, number of examples) so numbers are never shown without context.

- generation_accuracy: exact match of generated answers, with EOS error counts
- depth_curve / rounds_curve: the same metric over several T or S
- r_interventions: normal R vs zeroed R vs R shuffled between samples
- mlm_accuracy: masked token accuracy on held-out text
- retrieval_metrics: Recall@k and MRR for an embedding mode
- core_diagnostics: state norm, update norm, slot cosine and effective rank per step
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .corruption import mask_for_mlm, pad_batch
from .generation import generate
from .model import NuriModel

Example = tuple[list[int], list[int]]  # (prompt ids, answer ids without EOS)


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


@torch.no_grad()
def generation_accuracy(model: NuriModel, examples: Sequence[Example], steps: int | None = None,
                        rounds: int | None = None, oracle_length: bool = False, batch_size: int = 32,
                        transform_r: Callable[[torch.Tensor], torch.Tensor] | None = None) -> dict:
    """Exact match of generated answers.

    oracle_length=True gives the canvas the gold length (answer + EOS), which
    isolates denoising errors from length errors. Otherwise the length head
    picks the length, so the model must be instruction-tuned.
    """
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    counts = {"correct": 0, "missing_eos": 0, "wrong_length": 0, "total": 0}
    for chunk in _batches(list(examples), batch_size):
        prompts = [p[-config.max_input_len:] for p, _ in chunk]
        ids, valid = pad_batch(prompts, config.pad_id)
        gold_lengths = torch.tensor([min(len(a) + 1, config.max_response_len) for _, a in chunk])
        result = generate(model, ids.to(device), valid.to(device), steps=steps, rounds=rounds,
                          canvas_len=gold_lengths.to(device) if oracle_length else None,
                          transform_r=transform_r)
        for (_, answer), tokens, ended, length, gold in zip(chunk, result.tokens, result.ended_with_eos,
                                                            result.lengths.tolist(), gold_lengths.tolist()):
            counts["correct"] += tokens == list(answer)
            counts["missing_eos"] += not ended
            counts["wrong_length"] += length != gold
            counts["total"] += 1
    total = max(counts["total"], 1)
    return {"exact_match": counts["correct"] / total, "missing_eos_rate": counts["missing_eos"] / total,
            "length_error_rate": counts["wrong_length"] / total, "examples": counts["total"],
            "steps": steps if steps is not None else config.default_depth,
            "rounds": rounds if rounds is not None else config.refine_rounds, "oracle_length": oracle_length}


def depth_curve(model: NuriModel, examples: Sequence[Example], depths=(1, 2, 4, 8, 16), **kwargs) -> list[dict]:
    """Exact match at several reasoning depths T of the same checkpoint (D-014)."""
    return [generation_accuracy(model, examples, steps=t, **kwargs) for t in depths]


def rounds_curve(model: NuriModel, examples: Sequence[Example], rounds=(1, 2, 4, 8), **kwargs) -> list[dict]:
    """Exact match at several refinement rounds S."""
    return [generation_accuracy(model, examples, rounds=s, **kwargs) for s in rounds]


def zero_r(r: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(r)


def shuffle_r(r: torch.Tensor) -> torch.Tensor:
    """Give every sample the R of the next sample (a fixed derangement for batches > 1)."""
    return r.roll(1, dims=0)


def r_interventions(model: NuriModel, examples: Sequence[Example], **kwargs) -> dict:
    """If accuracy barely drops when R is removed or swapped, the readout is bypassing the core."""
    return {"normal": generation_accuracy(model, examples, **kwargs),
            "zero_r": generation_accuracy(model, examples, transform_r=zero_r, **kwargs),
            "shuffled_r": generation_accuracy(model, examples, transform_r=shuffle_r, **kwargs)}


@torch.no_grad()
def mlm_accuracy(model: NuriModel, documents: Sequence[list[int]], rate: float = 0.15, batch_size: int = 32,
                 seed: int = 0) -> dict:
    """Top-1 accuracy and loss on masked tokens of held-out documents (first N tokens of each)."""
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    generator = torch.Generator().manual_seed(seed)
    special = (config.pad_id, config.mask_id, config.eos_id)
    correct = targets = 0
    loss_sum = 0.0
    for chunk in _batches(list(documents), batch_size):
        ids, valid = pad_batch([d[:config.max_input_len] for d in chunk], config.pad_id)
        corrupted, mask = mask_for_mlm(ids, valid, config.mask_id, config.vocab_size, special, rate, generator)
        x, _ = model.encode(corrupted.to(device), valid.to(device))
        logits = model.mlm_logits(x, mask.to(device)).float()
        labels = ids[mask].to(device)
        correct += int((logits.argmax(-1) == labels).sum())
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        targets += int(mask.sum())
    return {"accuracy": correct / max(targets, 1), "loss": loss_sum / max(targets, 1), "targets": targets,
            "mask_rate": rate}


@torch.no_grad()
def retrieval_metrics(model: NuriModel, pairs: Sequence[Example], mode: str = "fast", steps: int | None = None,
                      dim: int | None = None, ks=(1, 5, 10), batch_size: int = 64) -> dict:
    """Each query should retrieve its own document among all documents in `pairs`."""
    model.eval()
    device = next(model.parameters()).device

    def embed(side: int) -> torch.Tensor:
        out = []
        for chunk in _batches([p[side][:model.config.max_input_len] for p in pairs], batch_size):
            ids, valid = pad_batch(chunk, model.config.pad_id)
            out.append(model.embed(ids.to(device), valid.to(device), mode=mode, steps=steps, dim=dim))
        return torch.cat(out)

    scores = embed(0) @ embed(1).T
    gold = scores.diagonal().unsqueeze(1)
    ranks = (scores > gold).sum(dim=1) + 1  # 1 = the right document scored highest
    result = {f"recall@{k}": float((ranks <= k).float().mean()) for k in ks}
    result.update(mrr=float((1.0 / ranks.float()).mean()), pairs=len(pairs), mode=mode, dim=dim, steps=steps)
    return result


@torch.no_grad()
def core_diagnostics(model: NuriModel, prompts: Sequence[list[int]], steps: int = 8) -> list[dict]:
    """Per-step statistics of the raw recurrent state, averaged over the batch.

    These are diagnostics only: a state that stops changing is not proof of
    collapse, and a changing state is not proof of reasoning (§12).
    """
    model.eval()
    device = next(model.parameters()).device
    ids, valid = pad_batch([p[:model.config.max_input_len] for p in prompts], model.config.pad_id)
    x, valid = model.encode(ids.to(device), valid.to(device))
    _, states = model.reason(x, valid, steps, return_states=True)
    rows = []
    for t, state in enumerate(states):
        state = state.float()
        slots = F.normalize(state, dim=-1)
        cosine = slots @ slots.transpose(1, 2)
        k = state.shape[1]
        off_diagonal = (cosine.sum(dim=(1, 2)) - k) / max(k * (k - 1), 1)
        singular = torch.linalg.svdvals(state - state.mean(dim=1, keepdim=True))
        p = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum(dim=-1))
        row = {"step": t, "state_norm": float(state.norm(dim=-1).mean()),
               "slot_cosine": float(off_diagonal.mean()), "effective_rank": float(effective_rank.mean())}
        if t > 0:
            update = (state - states[t - 1].float()).norm(dim=-1) / states[t - 1].float().norm(dim=-1).clamp_min(1e-12)
            row["relative_update"] = float(update.mean())
        rows.append(row)
    return rows


def sample_examples(examples: Sequence[Example], count: int, seed: int = 0) -> list[Example]:
    """A fixed random subset, so repeated evaluations use the same examples."""
    examples = list(examples)
    return random.Random(seed).sample(examples, min(count, len(examples)))
