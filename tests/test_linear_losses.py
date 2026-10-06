"""Check bounded vocabulary projections against the original losses and gradients."""

import pytest
import torch
import torch.nn.functional as F

from nuri.losses import (
    linear_sample_mean_ce,
    linear_token_mean_ce,
    sample_mean_ce,
    token_mean_ce,
)


@pytest.mark.parametrize("chunk_size", [1, 3, 256])
@pytest.mark.parametrize("sample_mean", [False, True])
def test_values_and_tied_weight_gradients(chunk_size, sample_mean):
    torch.manual_seed(7)
    labels = torch.randint(0, 19, (3, 5))
    mask = torch.tensor([[True] * 5, [True, False, False, False, False], [False] * 5])
    embedding = torch.randn(19, 8, requires_grad=True)
    reference_weight = embedding.detach().clone().requires_grad_()
    # The same weight is used in the input and output paths, as in Nuri.
    selected = labels[mask]
    hidden = F.embedding(selected, embedding) * 0.7
    reference_hidden = F.embedding(selected, reference_weight) * 0.7
    logits = F.linear(reference_hidden, reference_weight)
    if sample_mean:
        actual = linear_sample_mean_ce(hidden, embedding, labels, mask, chunk_size)
        expected = sample_mean_ce(logits, labels, mask)
        assert not torch.isclose(expected, token_mean_ce(logits, selected))
    else:
        actual = linear_token_mean_ce(hidden, embedding, selected, chunk_size)
        expected = token_mean_ce(logits, selected)
    torch.testing.assert_close(actual, expected)
    (actual * 0.37).backward()
    (expected * 0.37).backward()
    torch.testing.assert_close(
        embedding.grad, reference_weight.grad, atol=2e-6, rtol=2e-5
    )


def test_logit_storage_is_not_retained_and_chunks_are_bounded(monkeypatch):
    hidden = torch.randn(13, 7, requires_grad=True)
    weight = torch.randn(103, 7, requires_grad=True)
    targets = torch.arange(13)
    shapes, saved = [], []
    original = F.linear

    def tracked(*args, **kwargs):
        output = original(*args, **kwargs)
        shapes.append(tuple(output.shape))
        return output

    monkeypatch.setattr(F, "linear", tracked)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tuple(tensor.shape)) or tensor,
        lambda tensor: tensor,
    ):
        loss = linear_token_mean_ce(hidden, weight, targets, chunk_size=4)
    assert (13, 103) not in saved
    assert not any(len(shape) == 2 and shape[1] == 103 for shape in saved)
    loss.backward()
    assert len(shapes) == 8  # Four projections in forward, four recomputed in backward.
    assert all(shape[0] <= 4 and shape[1] == 103 for shape in shapes)


def test_no_grad_and_autocast_match_reference():
    hidden = torch.randn(9, 8)
    weight = torch.randn(31, 8)
    targets = torch.arange(9)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = linear_token_mean_ce(hidden, weight, targets, 4)
        expected = token_mean_ce(F.linear(hidden, weight), targets)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="positive integer"):
        linear_token_mean_ce(
            torch.randn(2, 3), torch.randn(5, 3), torch.arange(2), chunk_size
        )


def test_empty_targets_are_rejected():
    with pytest.raises(ValueError, match="at least one target"):
        linear_token_mean_ce(
            torch.empty(0, 3), torch.randn(5, 3), torch.empty(0, dtype=torch.long)
        )


def test_selected_row_and_mask_contracts():
    with pytest.raises(ValueError, match="one target"):
        linear_token_mean_ce(torch.randn(3, 4), torch.randn(7, 4), torch.arange(2))
    with pytest.raises(ValueError, match="boolean"):
        linear_sample_mean_ce(
            torch.randn(2, 4), torch.randn(7, 4), torch.zeros(2, 3), torch.ones(2, 3)
        )
