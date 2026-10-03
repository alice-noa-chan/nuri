"""Bidirectional masked-canvas readout over the fixed memory [X; R_final] (ARCHITECTURE.md §7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nuri.config import NuriConfig
from nuri.encoder import RopeTables
from nuri.layers import Attention, RMSNorm
from nuri.moe import build_ffn


@dataclass
class MemoryCache:
    """Per-layer memory K/V for one generation request (or one training forward)."""

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    mask: torch.Tensor  # [B, N + K], True where the memory may be attended


class ReadoutBlock(nn.Module):
    def __init__(self, config: NuriConfig, layer_number: int):
        super().__init__()
        d = config.hidden_size
        self.self_norm = RMSNorm(d, config.norm_eps)
        self.self_attn = Attention(d, config.num_heads, config.qk_norm, config.norm_eps)
        self.cross_norm = RMSNorm(d, config.norm_eps)
        self.cross_attn = Attention(d, config.num_heads, config.qk_norm, config.norm_eps)
        self.ffn_norm = RMSNorm(d, config.norm_eps)
        self.ffn = build_ffn(config, "readout", layer_number)

    def forward(self, c, canvas_valid, kv, memory_mask, rope):
        c = c + self.self_attn(self.self_norm(c), key_mask=canvas_valid, rope=rope)
        c = c + self.cross_attn(self.cross_norm(c), kv=kv, key_mask=memory_mask)
        return c + self.ffn(self.ffn_norm(c), canvas_valid)


class BidirectionalReadout(nn.Module):
    """Canvas positions see the current corrupted canvas, X and R; X and R never see the canvas.

    Canvas self-attention uses RoPE over output-internal positions; memory
    cross-attention uses none. Weights are shared across refinement rounds.
    """

    def __init__(self, config: NuriConfig):
        super().__init__()
        self.rope = RopeTables(config.head_dim, config.rope_theta)
        self.blocks = nn.ModuleList(ReadoutBlock(config, i + 1) for i in range(config.readout_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def prepare_memory(self, x: torch.Tensor, valid: torch.Tensor, r: torch.Tensor) -> MemoryCache:
        memory = torch.cat([x, r], dim=1)
        slot_mask = torch.ones(r.shape[:2], dtype=torch.bool, device=r.device)
        mask = torch.cat([valid, slot_mask], dim=1)
        return MemoryCache([block.cross_attn.project_kv(memory) for block in self.blocks], mask)

    def forward(self, canvas: torch.Tensor, canvas_valid: torch.Tensor, cache: MemoryCache) -> torch.Tensor:
        """canvas: embedded canvas tokens [B, L, D]. Returns normalized hidden states."""
        rope = self.rope(canvas.shape[1], canvas.device)
        for block, kv in zip(self.blocks, cache.kv):
            canvas = block(canvas, canvas_valid, kv, cache.mask, rope)
        return self.final_norm(canvas)
