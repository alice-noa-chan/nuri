"""Masking must respect configured special IDs and reject invalid policies early."""

import pytest
import torch

from nuri.corruption import corrupt_canvas, mask_for_mlm


def test_noncontiguous_special_ids_never_become_random_replacements():
    special = (2, 5, 9)
    ids = torch.zeros(1, 10000, dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    generator = torch.Generator().manual_seed(21)
    before_roll = torch.Generator().manual_seed(21)
    torch.randperm(ids.numel(), generator=before_roll)
    rolls = torch.rand(ids.shape, generator=before_roll)
    random_positions = (rolls >= 0.8) & (rolls < 0.9)
    corrupted, targets = mask_for_mlm(ids, valid, 5, 12, special, 1.0, generator)
    assert targets.all() and random_positions.sum() > 500
    actual = set(corrupted[random_positions].tolist())
    assert actual == set(range(12)) - set(special)


def test_default_mlm_random_draws_and_rng_state_are_preserved():
    ids = torch.arange(3, 203).reshape(2, 100)
    valid = torch.ones_like(ids, dtype=torch.bool)
    generator = torch.Generator().manual_seed(4)
    replay = torch.Generator().manual_seed(4)
    expected = ids.clone()
    selected = torch.zeros_like(valid)
    for row in range(2):
        selected[row, torch.randperm(100, generator=replay)[:30]] = True
    rolls = torch.rand(ids.shape, generator=replay)
    expected[selected & (rolls < 0.8)] = 1
    replacements = torch.randint(3, 256, ids.shape, generator=replay)
    random_positions = selected & (rolls >= 0.8) & (rolls < 0.9)
    expected[random_positions] = replacements[random_positions]
    actual, targets = mask_for_mlm(ids, valid, 1, 256, (0, 1, 2), generator=generator)
    assert torch.equal(actual, expected) and torch.equal(targets, selected)
    assert torch.equal(generator.get_state(), replay.get_state())


@pytest.mark.parametrize("rate", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_mlm_rates_fail_without_consuming_rng(rate):
    generator = torch.Generator().manual_seed(2)
    before = generator.get_state()
    with pytest.raises(ValueError, match="rate"):
        mask_for_mlm(
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            1,
            8,
            (0, 1, 2),
            rate,
            generator,
        )
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize("special", [(0, 1, 9), (-1,), (True,), (0, 1, 2, 3)])
def test_invalid_random_replacement_vocabulary(special):
    with pytest.raises(ValueError):
        mask_for_mlm(torch.tensor([[3]]), torch.tensor([[True]]), 1, 4, special)


@pytest.mark.parametrize(
    "options",
    [
        {"full_mask_prob": float("nan")},
        {"full_mask_prob": -1},
        {"full_mask_prob": 1.1},
        {"full_mask_prob": True},
        {"ratio_range": (0, 1)},
        {"ratio_range": (0.9, 0.1)},
        {"ratio_range": (0.1, float("inf"))},
        {"ratio_range": (0.1,)},
    ],
)
def test_invalid_canvas_policy_fails_without_consuming_rng(options):
    generator = torch.Generator().manual_seed(2)
    before = generator.get_state()
    with pytest.raises(ValueError):
        corrupt_canvas(
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            1,
            generator=generator,
            **options,
        )
    assert torch.equal(before, generator.get_state())
