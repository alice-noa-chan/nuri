"""Bidirectional masked-canvas readout over the fixed memory [X; R_final] (ARCHITECTURE.md §7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import NuriConfig
from .encoder import RopeTables
from .layers import Attention, RMSNorm, run_checkpointed
from .moe import build_ffn


@dataclass
class MemoryCache:
    """Per-layer memory K/V for one generation request (or one training forward)."""

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    mask: torch.Tensor  # [B, N + K], True where the memory may be attended


class MemoryPositions(nn.Module):
    """Learned absolute positions for the input part of the readout memory.

    X carries only relative (RoPE) position information and memory
    cross-attention has none, so without this a canvas position can find a
    token by its content but cannot read the input in order (see
    docs/GENERATION_DIAGNOSTICS.md). Each valid input token gets one vector
    for its index from the first token and one for its index back from the
    last token, so both copying and continuing from the end are direct
    lookups. Inputs are right-padded; padding gets no vector. Both tables
    start at zero, so a newly enabled model computes exactly what it did
    before until training changes them.
    """

    def __init__(self, max_input_len: int, hidden_size: int):
        super().__init__()
        self.from_start = nn.Embedding(max_input_len, hidden_size)
        self.from_end = nn.Embedding(max_input_len, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.from_start.weight)
        nn.init.zeros_(self.from_end.weight)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] input memory, valid: [B, N] real tokens (right-padded)."""
        if x.shape[1] > self.from_start.num_embeddings:
            raise ValueError("the input is longer than the memory position table")
        positions = torch.arange(x.shape[1], device=x.device)[None, :]
        from_end = valid.sum(dim=1, keepdim=True) - 1 - positions
        start_ids = positions.expand_as(valid) * valid
        end_ids = from_end.clamp(min=0) * valid
        vectors = self.from_start(start_ids) + self.from_end(end_ids)
        return x + vectors.to(x.dtype) * valid.unsqueeze(-1).to(x.dtype)


class ReadoutBlock(nn.Module):
    def __init__(self, config: NuriConfig, layer_number: int):
        super().__init__()
        d = config.hidden_size
        self.self_norm = RMSNorm(d, config.norm_eps)
        self.self_attn = Attention(d, config.num_heads, config.qk_norm, config.norm_eps, config.num_kv_heads, config.kv_latent_dim)
        self.cross_norm = RMSNorm(d, config.norm_eps)
        self.cross_attn = Attention(d, config.num_heads, config.qk_norm, config.norm_eps, config.num_kv_heads, config.kv_latent_dim)
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
        self.max_response_len = config.max_response_len
        self.position_embedding = (nn.Embedding(config.max_response_len, config.hidden_size)
                                   if config.canvas_position_encoding == "learned" else None)
        self.memory_positions = (MemoryPositions(config.max_input_len, config.hidden_size)
                                 if config.memory_position_encoding == "learned" else None)
        self.gradient_checkpointing = False  # set by NuriModel.set_gradient_checkpointing
        self.blocks = nn.ModuleList(ReadoutBlock(config, i + 1) for i in range(config.readout_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def prepare_memory(self, x: torch.Tensor, valid: torch.Tensor, r: torch.Tensor) -> MemoryCache:
        if self.memory_positions is not None:
            x = self.memory_positions(x, valid)
        memory = torch.cat([x, r], dim=1)
        slot_mask = torch.ones(r.shape[:2], dtype=torch.bool, device=r.device)
        mask = torch.cat([valid, slot_mask], dim=1)
        return MemoryCache([block.cross_attn.project_kv(memory) for block in self.blocks], mask)

    def forward(self, canvas: torch.Tensor, canvas_valid: torch.Tensor, cache: MemoryCache) -> torch.Tensor:
        """canvas: embedded canvas tokens [B, L, D]. Returns normalized hidden states."""
        if len(cache.kv) != len(self.blocks):
            raise ValueError(f"memory cache has {len(cache.kv)} layers, the readout has {len(self.blocks)}")
        length = canvas.shape[1]
        if not 1 <= length <= self.max_response_len:
            raise ValueError(f"canvas length must be in 1..{self.max_response_len}")
        if self.position_embedding is not None:
            # Q/K-only RoPE cannot distinguish an all-MASK canvas whose V
            # vectors are identical. Put position information in the residual
            # and value path once per fresh readout, including the first fill.
            positions = torch.arange(length, device=canvas.device)
            position_values = self.position_embedding(positions).to(canvas.dtype)
            canvas = canvas + position_values * canvas_valid.unsqueeze(-1)
        rope = self.rope(canvas.shape[1], canvas.device)
        for block, kv in zip(self.blocks, cache.kv):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                canvas = run_checkpointed(block, canvas, canvas_valid, kv, cache.mask, rope)
            else:
                canvas = block(canvas, canvas_valid, kv, cache.mask, rope)
        return self.final_norm(canvas)
