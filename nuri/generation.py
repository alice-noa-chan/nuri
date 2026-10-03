"""Mask-Predict style refinement sampler (ARCHITECTURE.md §7).

X and R are computed once per request. Each round fills every MASK position
in parallel, then re-masks the least confident positions. After the last
round the answer is cut at the first EOS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from nuri.model import NuriModel


@dataclass
class GenerationResult:
    tokens: list[list[int]]          # answers, cut before the first EOS
    canvas: torch.Tensor             # final canvas [B, L] (padding = pad_id)
    lengths: torch.Tensor            # canvas length used for each sample
    ended_with_eos: list[bool]       # False means the model never wrote EOS
    history: list[torch.Tensor] = field(default_factory=list)  # canvas after each round


@torch.no_grad()
def generate(model: NuriModel, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
             steps: int | None = None, rounds: int | None = None, canvas_len=None,
             max_len: int | None = None, keep_history: bool = False) -> GenerationResult:
    """Greedy refinement generation.

    steps: reasoning depth T (config default if None).
    rounds: refinement rounds S (config default if None).
    canvas_len: an int or a [B] tensor of output lengths including EOS. Base
        models must pass it; instruction-tuned models may omit it and use the
        length head.
    max_len: upper bound for the canvas length (at most M).
    """
    config = model.config
    steps = config.default_depth if steps is None else steps
    rounds = config.refine_rounds if rounds is None else rounds
    max_len = min(max_len or config.max_response_len, config.max_response_len)

    x, valid = model.encode(input_ids, attention_mask)
    r = model.reason(x, valid, steps)
    lengths = _choose_lengths(model, x, valid, r, canvas_len, input_ids.shape[0]).clamp(1, max_len)
    cache = model.generation_memory(x, valid, r)

    batch, width = input_ids.shape[0], int(lengths.max())
    positions = torch.arange(width, device=input_ids.device)
    canvas_valid = positions[None, :] < lengths[:, None]
    canvas = torch.full((batch, width), config.mask_id, device=input_ids.device)
    canvas[~canvas_valid] = config.pad_id
    confidence = torch.zeros(batch, width, device=input_ids.device)
    history = []

    for round_index in range(1, rounds + 1):
        masked = (canvas == config.mask_id) & canvas_valid
        hidden = model.readout_hidden(canvas, canvas_valid, cache)
        logits = model.vocab_logits(hidden[masked]).float()
        # MASK and PAD are not words; never write them into the answer.
        logits[:, [config.mask_id, config.pad_id]] = float("-inf")
        probs = logits.softmax(dim=-1)
        best_prob, best_token = probs.max(dim=-1)
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
    return GenerationResult(tokens, canvas, lengths, ended, history)


def _choose_lengths(model, x, valid, r, canvas_len, batch) -> torch.Tensor:
    if canvas_len is not None:
        lengths = torch.as_tensor(canvas_len, device=x.device).long()
        return lengths.expand(batch).clone() if lengths.dim() == 0 else lengths
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
