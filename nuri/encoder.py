"""Bidirectional input encoder producing the read-only memory X (ARCHITECTURE.md §4)."""

from __future__ import annotations

import torch
from torch import nn

from .config import NuriConfig
from .layers import Attention, RMSNorm, rope_cache, run_checkpointed
from .moe import build_ffn


class RopeTables(nn.Module):
    """Lazily grown cos/sin tables shared by every RoPE user in one stack."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim, self.theta = head_dim, theta
        self._tables = None

    def forward(self, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tables is None or self._tables[0].shape[0] < length or self._tables[0].device != torch.device(device):
            self._tables = rope_cache(max(length, 64), self.head_dim, self.theta, device)
        return self._tables


def local_mask(valid: torch.Tensor, radius: int) -> torch.Tensor:
    """[B, N, N] mask: query i may attend to valid key j when |i - j| <= radius.

    Every position may also attend to itself. Without that, a padding query far
    from any real token would have no key at all, and SDPA returns NaN for
    such a row, which would then leak into later layers.
    """
    n = valid.shape[1]
    positions = torch.arange(n, device=valid.device)
    band = (positions[:, None] - positions[None, :]).abs() <= radius
    itself = torch.eye(n, dtype=torch.bool, device=valid.device)
    return (band[None] & valid[:, None, :]) | itself[None]


def segment_mask(segments: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B, N, N] mask for packed rows: a token sees only valid tokens of its own document.

    segments: [B, N] document number of each token (padding may hold any value).
    Every position may also see itself, so no attention row is ever empty.
    """
    same = segments[:, :, None] == segments[:, None, :]
    itself = torch.eye(segments.shape[1], dtype=torch.bool, device=segments.device)
    return (same & valid[:, None, :]) | itself[None]


class EncoderBlock(nn.Module):
    def __init__(self, config: NuriConfig, layer_number: int):
        super().__init__()
        schedule = config.attention_schedule
        self.local = bool(schedule) and schedule[layer_number - 1] == "L"
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = Attention(config.hidden_size, config.num_heads, config.qk_norm, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn = build_ffn(config, "input", layer_number)

    def forward(self, h, valid, rope, local_keys=None, global_keys=None):
        """global_keys: optional per-query mask (packed rows); otherwise the padding mask is used."""
        keys = local_keys if self.local else (global_keys if global_keys is not None else valid)
        h = h + self.attn(self.attn_norm(h), key_mask=keys, rope=rope)
        return h + self.ffn(self.ffn_norm(h), valid)


class InputEncoder(nn.Module):
    """X = E_phi(tokens).

    It takes already embedded tokens. The token embedding lives in the model,
    which shares it with the canvas input and the vocabulary head; keeping it
    out of this module means the weight has exactly one name in checkpoints.
    """

    def __init__(self, config: NuriConfig):
        super().__init__()
        self.local_radius = config.local_radius
        self.gradient_checkpointing = False  # set by NuriModel.set_gradient_checkpointing
        self.rope = RopeTables(config.head_dim, config.rope_theta)
        self.blocks = nn.ModuleList(EncoderBlock(config, i + 1) for i in range(config.input_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, embedded: torch.Tensor, valid: torch.Tensor,
                segments: torch.Tensor | None = None) -> torch.Tensor:
        """embedded: [B, N, D] token embeddings; valid: [B, N] real-token mask.

        segments: optional [B, N] document numbers for packed rows (several
        documents in one row). Tokens then attend only within their own
        document. RoPE is relative, so a document's offset in the row does
        not matter.
        """
        h = embedded
        rope = self.rope(h.shape[1], h.device)
        # Masks are built once per forward and shared by all layers.
        global_keys = segment_mask(segments, valid) if segments is not None else None
        local_keys = None
        if any(b.local for b in self.blocks):
            local_keys = local_mask(valid, self.local_radius)
            if global_keys is not None:
                local_keys = local_keys & global_keys
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = run_checkpointed(block, h, valid, rope, local_keys, global_keys)
            else:
                h = block(h, valid, rope, local_keys, global_keys)
        return self.final_norm(h)
