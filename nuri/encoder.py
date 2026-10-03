"""Bidirectional input encoder producing the read-only memory X (ARCHITECTURE.md §4)."""

from __future__ import annotations

import torch
from torch import nn

from nuri.config import NuriConfig
from nuri.layers import Attention, RMSNorm, rope_cache
from nuri.moe import build_ffn


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


class EncoderBlock(nn.Module):
    def __init__(self, config: NuriConfig, layer_number: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = Attention(config.hidden_size, config.num_heads, config.qk_norm, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn = build_ffn(config, "input", layer_number)

    def forward(self, h, valid, rope):
        h = h + self.attn(self.attn_norm(h), key_mask=valid, rope=rope)
        return h + self.ffn(self.ffn_norm(h), valid)


class InputEncoder(nn.Module):
    """X = E_phi(tokens). The token embedding is owned by the model and shared."""

    def __init__(self, config: NuriConfig, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
        self.rope = RopeTables(config.head_dim, config.rope_theta)
        self.blocks = nn.ModuleList(EncoderBlock(config, i + 1) for i in range(config.input_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, input_ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.embedding(input_ids)
        rope = self.rope(input_ids.shape[1], input_ids.device)
        for block in self.blocks:
            h = block(h, valid, rope)
        return self.final_norm(h)
