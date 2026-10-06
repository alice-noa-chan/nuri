"""Training losses (ARCHITECTURE.md §6, §7, §8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _linear_ce_sum(hidden, weight, targets, token_weights):
    logits = F.linear(hidden, weight)
    losses = F.cross_entropy(logits.float(), targets, reduction="none")
    return (losses * token_weights).sum()


def _chunked_linear_ce(hidden, weight, targets, token_weights, chunk_size):
    """Recompute each vocabulary projection in backward instead of retaining logits."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and vocabulary weight must be compatible matrices")
    if targets.ndim != 1 or targets.shape[0] != hidden.shape[0] or not targets.numel():
        raise ValueError(
            "each selected hidden row needs one target, with at least one target"
        )
    if token_weights.shape != targets.shape:
        raise ValueError("token weights must have the same shape as targets")

    loss = hidden.new_zeros((), dtype=torch.float32)
    recompute = torch.is_grad_enabled() and (
        hidden.requires_grad or weight.requires_grad
    )
    for start in range(0, hidden.shape[0], chunk_size):
        arguments = (
            hidden[start : start + chunk_size],
            weight,
            targets[start : start + chunk_size],
            token_weights[start : start + chunk_size],
        )
        if recompute:
            part = checkpoint(
                _linear_ce_sum,
                *arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            part = _linear_ce_sum(*arguments)
        loss = loss + part
    return loss


def linear_token_mean_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """MLM token mean with bounded transient logits and ordinary tied-weight gradients.

    Only selected positions are supplied. Checkpointing the loss projection
    bounds its logits to chunk_size x vocabulary_size during both forward and
    backward. This trades recomputation for memory; measure actual GPU speed.
    """
    token_weights = torch.full_like(
        targets, 1 / max(targets.numel(), 1), dtype=torch.float32
    )
    return _chunked_linear_ce(hidden, weight, targets, token_weights, chunk_size)


def linear_sample_mean_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Canvas row mean with the same weighting as sample_mean_ce, without full logits.

    hidden contains loss-bearing positions in row-major order. Each target
    gets weight 1 / (its row's target count x number of target-bearing rows).
    Empty rows are excluded, so different answer lengths retain equal weight.
    """
    if (
        loss_mask.ndim != 2
        or loss_mask.dtype != torch.bool
        or targets.shape != loss_mask.shape
    ):
        raise ValueError(
            "targets and a boolean two-dimensional loss_mask must have the same shape"
        )
    counts = loss_mask.sum(dim=1)
    rows = loss_mask.nonzero(as_tuple=True)[0]
    target_rows = (counts > 0).sum()
    token_weights = counts[rows].float().reciprocal() / target_rows.clamp_min(1)
    return _chunked_linear_ce(
        hidden, weight, targets[loss_mask], token_weights, chunk_size
    )


def token_mean_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain cross-entropy averaged over the given tokens (used for MLM)."""
    return F.cross_entropy(logits.float(), targets)


def sample_mean_ce(
    logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Canvas denoising loss (§7): average CE inside each sample, then over the batch.

    logits: [num_masked, V] computed only at `loss_mask` positions (row-major
    order). This differs from a token mean: a short response counts as much as
    a long one.
    """
    ce = F.cross_entropy(logits.float(), targets[loss_mask], reduction="none")
    rows = loss_mask.nonzero(as_tuple=True)[0]
    per_sample = torch.zeros(loss_mask.shape[0], device=ce.device).index_add_(
        0, rows, ce
    )
    counts = loss_mask.sum(dim=1)
    has_target = counts > 0
    return (per_sample[has_target] / counts[has_target]).mean()


def length_loss(length_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """lengths count EOS and are in 1..M; class index = length - 1."""
    return F.cross_entropy(length_logits.float(), lengths - 1)


def contrastive_loss(
    query: torch.Tensor,
    document: torch.Tensor,
    temperature: float = 0.05,
    dims: tuple[int, ...] | None = None,
    false_negatives: torch.Tensor | None = None,
) -> torch.Tensor:
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
            scores = scores.masked_fill(
                false_negatives.to(scores.device), float("-inf")
            )
        total = total + 0.5 * (
            F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)
        )
    return total / len(dims)
