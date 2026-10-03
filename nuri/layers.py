"""Shared building blocks: RMSNorm, RoPE, attention, SwiGLU (ARCHITECTURE.md §4, §9)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cache(length: int, head_dim: int, theta: float, device, dtype=torch.float32):
    """cos/sin tables of shape [length, head_dim] for rotate-half RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x: [B, H, n, head_dim]; rope tables cover at least n positions."""
    cos, sin = rope
    n = x.shape[-2]
    cos, sin = cos[:n].to(x.dtype), sin[:n].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    """Bidirectional multi-head attention used for self- and cross-attention.

    Order per §9: projection → per-head QK RMSNorm → RoPE (only where the
    caller passes tables) → SDPA with scale 1/sqrt(head_dim). Key/value
    projections can be computed once with `project_kv` and reused, e.g. the
    fixed input memory across all reasoning steps or refinement rounds.
    """

    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps) if qk_norm else None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, memory: torch.Tensor, rope=None) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._split(self.k_proj(memory))
        if self.k_norm is not None:
            k = self.k_norm(k)
        if rope is not None:
            k = apply_rope(k, rope)
        return k, self._split(self.v_proj(memory))

    def forward(self, x, kv=None, key_mask=None, rope=None) -> torch.Tensor:
        """x: queries [B, n, D]. kv: cached (k, v) or None for self-attention.

        key_mask: bool [B, n_keys], True for keys that may be attended.
        rope: tables applied to queries (and keys when kv is None).
        """
        q = self._split(self.q_proj(x))
        if self.q_norm is not None:
            q = self.q_norm(q)
        if rope is not None:
            q = apply_rope(q, rope)
        k, v = kv if kv is not None else self.project_kv(x, rope)
        attn_mask = key_mask[:, None, None, :] if key_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        b, _, n, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        # `valid` is accepted so dense and MoE FFNs share one call signature.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
