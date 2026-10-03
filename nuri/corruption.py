"""Turn clean token sequences into training inputs (ARCHITECTURE.md §6, §7, §8-C).

Three corruptions are used:
- MLM masking of the input tokens (base stage, trains X directly).
- Canvas corruption of a response (base span denoising and IT).
- Splitting a raw document into an input prefix and a target span (base stage).
"""

from __future__ import annotations

import random

import torch


def pad_batch(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token lists. Returns ids [B, L] and a bool mask of real tokens."""
    length = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    valid = torch.zeros((len(sequences), length), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        valid[row, :len(seq)] = True
    return ids, valid


def _random_positions(valid_row: torch.Tensor, count: int, generator) -> torch.Tensor:
    """Pick `count` distinct positions among the True entries of one row."""
    positions = valid_row.nonzero(as_tuple=True)[0]
    order = torch.randperm(positions.numel(), generator=generator)
    return positions[order[:count]]


def mask_for_mlm(ids: torch.Tensor, valid: torch.Tensor, mask_id: int, vocab_size: int,
                 special_ids: tuple[int, ...], rate: float = 0.3, generator=None):
    """BERT-style masking: pick `rate` of real tokens; 80% -> MASK, 10% random, 10% kept.

    Every sample gets at least one target. Returns (corrupted ids, target mask).
    The 30% rate and 80/10/10 split are implementation choices; the spec does
    not fix them.
    """
    corrupted = ids.clone()
    targets = torch.zeros_like(valid)
    maskable = valid.clone()
    for special in special_ids:
        maskable &= ids != special
    for row in range(ids.shape[0]):
        available = int(maskable[row].sum())
        if available == 0:
            continue
        count = max(1, round(rate * available))
        targets[row, _random_positions(maskable[row], count, generator)] = True

    roll = torch.rand(ids.shape, generator=generator)
    to_mask = targets & (roll < 0.8)
    to_random = targets & (roll >= 0.8) & (roll < 0.9)
    corrupted[to_mask] = mask_id
    random_ids = torch.randint(len(special_ids), vocab_size, ids.shape, generator=generator)
    corrupted[to_random] = random_ids[to_random]
    return corrupted, targets


def corrupt_canvas(targets: torch.Tensor, valid: torch.Tensor, mask_id: int, full_mask_prob: float = 0.1,
                   ratio_range: tuple[float, float] = (0.15, 0.95), generator=None):
    """Mask part of each response canvas (§7).

    With probability `full_mask_prob` the whole response is masked; otherwise a
    ratio is drawn uniformly from `ratio_range`. EOS is a normal target. At
    least one position is always masked. Masked positions hold the real MASK
    id, so the answer never stays in the input. Returns (canvas ids, loss mask).
    """
    canvas = targets.clone()
    loss_mask = torch.zeros_like(valid)
    low, high = ratio_range
    for row in range(targets.shape[0]):
        length = int(valid[row].sum())
        if length == 0:
            continue
        if torch.rand(1, generator=generator).item() < full_mask_prob:
            count = length
        else:
            ratio = low + (high - low) * torch.rand(1, generator=generator).item()
            count = max(1, round(ratio * length))
        loss_mask[row, _random_positions(valid[row], count, generator)] = True
    canvas[loss_mask] = mask_id
    return canvas, loss_mask


def split_span(document: list[int], max_input_len: int, max_response_len: int, eos_id: int,
               min_span: int = 16, rng: random.Random | None = None) -> tuple[list[int], list[int]]:
    """Cut one raw document into (input prefix, target span) for text span denoising (§8-C).

    - The cut point c leaves at least one token on each side.
    - The input is the up to `max_input_len` tokens right before c.
    - The span starts at c and has between `min_span` and `max_response_len`
      tokens (fewer if the document is short).
    - EOS is appended only when the span reaches the end of the document,
      because only then is it a real ending. If such a span already fills the
      whole canvas, the cut moves forward so the span keeps ending at the
      document end and EOS still fits.
    """
    if len(document) < 2:
        raise ValueError("a document needs at least two tokens to be split")
    rng = rng or random
    cut = rng.randint(1, len(document) - 1)
    remaining = len(document) - cut
    longest = min(max_response_len, remaining)
    span = rng.randint(min(min_span, longest), longest)
    if cut + span == len(document):
        if span == max_response_len:
            cut += 1
            span -= 1
        target = document[cut:cut + span] + [eos_id]
    else:
        target = document[cut:cut + span]
    source = document[max(0, cut - max_input_len):cut]
    return source, target
