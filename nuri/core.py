"""Recurrent latent reasoning core (ARCHITECTURE.md §5)."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import NuriConfig
from .layers import Attention, RMSNorm
from .moe import MoEFFN, build_ffn


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
        self.ffn = build_ffn(config, "core", 1)  # dense unless MoEConfig.core is on
        # Raw logits of the three sigmoid branch gates (cross, self, ffn).
        gate_logit = math.log(config.gate_init / (1 - config.gate_init))
        self.gate_logits = nn.Parameter(torch.full((3,), gate_logit))
        self.final_norm = RMSNorm(d, config.norm_eps)
        nn.init.normal_(self.slots, std=config.initializer_range)
        # Steps each sample of the last forward actually ran ([B]); differs from
        # `steps` only when halting is on.
        self.last_steps_used: torch.Tensor | None = None
        # With a MoE core: expert assignment counts of every step of the last forward.
        self.last_step_expert_counts: list[torch.Tensor] = []

    def _step(self, r, kv, valid, gates):
        g_cross, g_self, g_ffn = gates
        r = r + g_cross * self.cross_attn(self.cross_norm(r), kv=kv, key_mask=valid)
        r = r + g_self * self.self_attn(self.self_norm(r))
        r = r + g_ffn * self.ffn(self.ffn_norm(r))
        if isinstance(self.ffn, MoEFFN) and self.ffn.last_aux is not None:
            self.last_step_expert_counts.append(self.ffn.last_aux["counts"])
        return r

    def forward(self, x: torch.Tensor, valid: torch.Tensor, steps: int, return_states: bool = False,
                halt_tol: float | None = None):
        """x: [B, N, D] input memory, valid: [B, N]. Returns R_final = Norm(R_T).

        With return_states, also returns the raw states [R_0, ..., R_T] for
        diagnostics (state norm, update norm, slot rank).

        halt_tol (experimental, inference): a sample stops updating once one
        step changes its state by less than halt_tol relative to its size
        (||R_new - R|| / ||R||). Stopped samples leave the batch, so they cost
        no more compute; `steps` is the upper bound. Training-free; learned
        halting is a separate experiment (§9).
        """
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if halt_tol is not None and (return_states or halt_tol < 0):
            raise ValueError("halt_tol must be >= 0 and cannot be combined with return_states")
        kv = self.cross_attn.project_kv(x)
        gates = torch.sigmoid(self.gate_logits).to(x.dtype)
        self.last_step_expert_counts = []
        r = self.slots.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        used = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if halt_tol is None:
            states = [r] if return_states else None
            for _ in range(steps):
                r = self._step(r, kv, valid, gates)
                if return_states:
                    states.append(r)
            self.last_steps_used = used + steps
            out = self.final_norm(r)
            return (out, states) if return_states else out

        active = torch.arange(x.shape[0], device=x.device)
        for _ in range(steps):
            if active.numel() == 0:
                break
            before = r[active]
            after = self._step(before, (kv[0][active], kv[1][active]), valid[active], gates)
            r = r.index_copy(0, active, after)
            used[active] += 1
            change = (after - before).flatten(1).norm(dim=1) / before.flatten(1).norm(dim=1).clamp_min(1e-12)
            active = active[change >= halt_tol]
        self.last_steps_used = used
        return self.final_norm(r)
