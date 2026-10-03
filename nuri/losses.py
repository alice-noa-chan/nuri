"""Training losses (ARCHITECTURE.md §6, §7, §8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_mean_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain cross-entropy averaged over the given tokens (used for MLM)."""
    return F.cross_entropy(logits.float(), targets)


def sample_mean_ce(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Canvas denoising loss (§7): average CE inside each sample, then over the batch.

    logits: [num_masked, V] computed only at `loss_mask` positions (row-major
    order). This differs from a token mean: a short response counts as much as
    a long one.
    """
    ce = F.cross_entropy(logits.float(), targets[loss_mask], reduction="none")
    rows = loss_mask.nonzero(as_tuple=True)[0]
    per_sample = torch.zeros(loss_mask.shape[0], device=ce.device).index_add_(0, rows, ce)
    counts = loss_mask.sum(dim=1)
    has_target = counts > 0
    return (per_sample[has_target] / counts[has_target]).mean()


def length_loss(length_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """lengths count EOS and are in 1..M; class index = length - 1."""
    return F.cross_entropy(length_logits.float(), lengths - 1)


def contrastive_loss(query: torch.Tensor, document: torch.Tensor, temperature: float = 0.05,
                     dims: tuple[int, ...] | None = None,
                     false_negatives: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric in-batch InfoNCE on unnormalized projections.

    Row i of `document` is the positive for row i of `query`; all other rows
    are negatives, except where false_negatives[i, j] is True (duplicates of
    the positive), which are left out of the softmax. With `dims`
    (Matryoshka), the loss is averaged over the prefixes z[:, :d], each
    renormalized.
    """
    dims = dims or (query.shape[-1],)
    labels = torch.arange(query.shape[0], device=query.device)
    total = 0.0
    for d in dims:
        q = F.normalize(query[:, :d].float(), dim=-1)
        k = F.normalize(document[:, :d].float(), dim=-1)
        scores = q @ k.T / temperature
        if false_negatives is not None:
            # The mask is symmetric, so it also fits the document-to-query direction.
            scores = scores.masked_fill(false_negatives.to(scores.device), float("-inf"))
        total = total + 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))
    return total / len(dims)
