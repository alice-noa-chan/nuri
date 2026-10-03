"""Mask-Predict style refinement sampler (ARCHITECTURE.md §7).

X and R are computed once per request. Each round fills every MASK position
in parallel, then re-masks the least confident positions. After the last
round the answer is cut at the first EOS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from .model import NuriModel


@dataclass
class GenerationResult:
    tokens: list[list[int]]          # answers, cut before the first EOS
    canvas: torch.Tensor             # final canvas [B, L] (padding = pad_id)
    lengths: torch.Tensor            # canvas length used for each sample
    ended_with_eos: list[bool]       # False means the model never wrote EOS
    history: list[torch.Tensor] = field(default_factory=list)  # canvas after each round
    reasoning_steps: torch.Tensor | None = None  # core steps each sample ran (< T only with halting)


@torch.no_grad()
def generate(model: NuriModel, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
             steps: int | None = None, rounds: int | None = None, canvas_len=None,
             max_len: int | None = None, keep_history: bool = False,
             transform_r: Callable[[torch.Tensor], torch.Tensor] | None = None,
             temperature: float = 0.0, top_k: int | None = None,
             generator: torch.Generator | None = None, halt_tol: float | None = None) -> GenerationResult:
    """Greedy refinement generation.

    steps: reasoning depth T (config default if None).
    rounds: refinement rounds S (config default if None).
    canvas_len: an int or a [B] tensor of output lengths including EOS. Base
        models must pass it; instruction-tuned models may omit it and use the
        length head.
    max_len: upper bound for the canvas length (at most M).
    transform_r: optional function applied to R_final before it is used, for
        intervention experiments (for example zeroing or shuffling R, §12).
        It affects both the length head and the readout memory.
    temperature: 0 (default) picks the most likely token. Above 0, tokens are
        sampled from softmax(logits / temperature), optionally restricted to
        the top_k most likely ones; pass a torch.Generator to make it
        repeatable. The stored confidence is the model's own probability of
        the chosen token (at temperature 1), so re-masking still prefers to
        revisit unlikely choices.
    halt_tol: experimental training-free halting of the core (see
        RecurrentReasoningCore); `steps` becomes an upper bound and
        result.reasoning_steps reports what each sample used.
    """
    config = model.config
    steps = config.default_depth if steps is None else steps
    rounds = config.refine_rounds if rounds is None else rounds
    max_len = config.max_response_len if max_len is None else max_len
    _check_positive_int(rounds, "rounds")
    _check_positive_int(max_len, "max_len")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None:
        _check_positive_int(top_k, "top_k")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be an int >= 0")
    max_len = min(max_len, config.max_response_len)

    x, valid = model.encode(input_ids, attention_mask)
    # halt_tol is passed only when used, so models that override reason() keep working.
    r = model.reason(x, valid, steps) if halt_tol is None else model.reason(x, valid, steps, halt_tol=halt_tol)
    reasoning_steps = model.core.last_steps_used
    if transform_r is not None:
        r = transform_r(r)
    lengths = _choose_lengths(model, x, valid, r, canvas_len, input_ids.shape[0]).clamp(1, max_len)
    cache = model.generation_memory(x, valid, r)
    draft_feedback = model.config.draft_feedback

    batch, width = input_ids.shape[0], int(lengths.max())
    positions = torch.arange(width, device=input_ids.device)
    canvas_valid = positions[None, :] < lengths[:, None]
    canvas = torch.full((batch, width), config.mask_id, device=input_ids.device)
    canvas[~canvas_valid] = config.pad_id
    confidence = torch.zeros(batch, width, device=input_ids.device)
    history = []

    for round_index in range(1, rounds + 1):
        masked = (canvas == config.mask_id) & canvas_valid
        if draft_feedback:  # experimental: R re-reads the current draft every round
            r = model.reason(x, valid, steps, draft=(canvas, canvas_valid))
            if transform_r is not None:
                r = transform_r(r)
            cache = model.generation_memory(x, valid, r)
        hidden = model.readout_hidden(canvas, canvas_valid, cache)
        logits = model.vocab_logits(hidden[masked]).float()
        # MASK and PAD are not words; never write them into the answer.
        logits[:, [config.mask_id, config.pad_id]] = float("-inf")
        probs = logits.softmax(dim=-1)
        best_token = _pick_tokens(logits, temperature, top_k, generator)
        best_prob = probs.gather(1, best_token[:, None]).squeeze(1)
        canvas[masked] = best_token
        # A token keeps the confidence it had when it was predicted from MASK;
        # we never re-score a visible token that can see itself.
        confidence[masked] = best_prob
        if round_index < rounds:
            _remask_least_confident(canvas, confidence, canvas_valid, lengths, round_index, rounds, config.mask_id)
        if keep_history:
            history.append(canvas.clone())

    tokens, ended = [], []
    for row in range(batch):
        answer = canvas[row, :int(lengths[row])].tolist()
        ended.append(config.eos_id in answer)
        tokens.append(answer[:answer.index(config.eos_id)] if config.eos_id in answer else answer)
    return GenerationResult(tokens, canvas, lengths, ended, history, reasoning_steps)


def _pick_tokens(logits: torch.Tensor, temperature: float, top_k: int | None, generator) -> torch.Tensor:
    """Greedy at temperature 0, otherwise a sample from the (top-k) tempered distribution."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    if top_k is not None and top_k < scaled.shape[-1]:
        cutoff = scaled.topk(top_k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    return torch.multinomial(scaled.softmax(dim=-1), 1, generator=generator).squeeze(1)


def _check_positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


def _explicit_lengths(canvas_len, batch: int, device) -> torch.Tensor:
    """Turn an int or a [B] integer tensor into a [B] length tensor, or fail clearly."""
    if isinstance(canvas_len, torch.Tensor):
        if canvas_len.is_floating_point() or canvas_len.is_complex() or canvas_len.dim() > 1:
            raise ValueError("canvas_len tensor must hold integers and be a scalar or a [batch] vector")
        lengths = canvas_len.to(device).long()
    else:
        _check_positive_int(canvas_len, "canvas_len")
        lengths = torch.tensor(canvas_len, device=device)
    if lengths.dim() == 0:
        lengths = lengths.expand(batch).clone()
    if lengths.shape[0] != batch:
        raise ValueError(f"canvas_len has {lengths.shape[0]} entries for a batch of {batch}")
    if (lengths < 1).any():
        raise ValueError("every canvas_len must be >= 1")
    return lengths


def _choose_lengths(model, x, valid, r, canvas_len, batch) -> torch.Tensor:
    if canvas_len is not None:
        return _explicit_lengths(canvas_len, batch, x.device)
    if not model.config.length_head_trained:
        raise ValueError("this model's length head is untrained (base model); pass canvas_len")
    u, r_mean = model.pool(x, valid, r)
    return model.length_logits(u, r_mean).argmax(dim=-1) + 1


def _remask_least_confident(canvas, confidence, canvas_valid, lengths, round_index, rounds, mask_id):
    """After round s, re-mask ceil(L * (S - s) / S) positions with the lowest confidence."""
    ranking = confidence.masked_fill(~canvas_valid, float("inf"))
    order = ranking.argsort(dim=1)
    for row in range(canvas.shape[0]):
        count = math.ceil(int(lengths[row]) * (rounds - round_index) / rounds)
        canvas[row, order[row, :count]] = mask_id
