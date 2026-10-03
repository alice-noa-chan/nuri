"""Recurrent latent reasoning core (ARCHITECTURE.md §5)."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import NuriConfig
from .layers import Attention, RMSNorm, SwiGLU


class RecurrentReasoningCore(nn.Module):
    """K learned slots updated T times by one shared block that reads X.

    A_t     = R_t + g_cross * CrossAttn(Norm(R_t), K_X, V_X)
    B_t     = A_t + g_self  * SelfAttn(Norm(A_t))
    R_(t+1) = B_t + g_ffn   * SwiGLU(Norm(B_t))

    K_X/V_X are projected once per forward and reused by every step with the
    autograd graph intact. Slots carry no RoPE; step embeddings are off.
    """

    def __init__(self, config: NuriConfig):
        super().__init__()
        d = config.hidden_size
        self.slots = nn.Parameter(torch.empty(config.latent_slots, d))
        self.cross_norm = RMSNorm(d, config.norm_eps)
        self.cross_attn = Attention(d, config.num_heads, config.qk_norm, config.norm_eps)
        self.self_norm = RMSNorm(d, config.norm_eps)
        self.self_attn = Attention(d, config.num_heads, config.qk_norm, config.norm_eps)
        self.ffn_norm = RMSNorm(d, config.norm_eps)
        self.ffn = SwiGLU(d, config.ffn_size)
        # Raw logits of the three sigmoid branch gates (cross, self, ffn).
        gate_logit = math.log(config.gate_init / (1 - config.gate_init))
        self.gate_logits = nn.Parameter(torch.full((3,), gate_logit))
        self.final_norm = RMSNorm(d, config.norm_eps)
        nn.init.normal_(self.slots, std=config.initializer_range)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, steps: int, return_states: bool = False):
        """x: [B, N, D] input memory, valid: [B, N]. Returns R_final = Norm(R_T).

        With return_states, also returns the raw states [R_0, ..., R_T] for
        diagnostics (state norm, update norm, slot rank).
        """
        if steps < 0:
            raise ValueError("steps must be >= 0")
        kv = self.cross_attn.project_kv(x)
        g_cross, g_self, g_ffn = torch.sigmoid(self.gate_logits).to(x.dtype)
        r = self.slots.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        states = [r] if return_states else None
        for _ in range(steps):
            r = r + g_cross * self.cross_attn(self.cross_norm(r), kv=kv, key_mask=valid)
            r = r + g_self * self.self_attn(self.self_norm(r))
            r = r + g_ffn * self.ffn(self.ffn_norm(r))
            if return_states:
                states.append(r)
        out = self.final_norm(r)
        return (out, states) if return_states else out
