"""Token-choice sparse MoE FFN and the dense/MoE FFN factory (ARCHITECTURE.md §9, D-011)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import NuriConfig
from .layers import SwiGLU


class MoEFFN(nn.Module):
    """Routed SwiGLU experts plus always-on shared experts.

    Routing is dropless and computed in fp32. Padding tokens are neither
    routed nor counted in the load statistics. Every training-mode forward
    appends one record (balance loss, router z-loss, per-expert assignment counts) to
    `aux_records`. A training step can run the same layer several times
    (e.g. query and document in embedding training), so records pile up
    until the model collects and clears them with `pop_moe_aux()`.
    """

    def __init__(self, hidden_size: int, num_experts: int, expert_ffn_size: int, top_k: int,
                 num_shared_experts: int):
        super().__init__()
        if not 2 <= top_k <= num_experts:
            # With top_k=1 the renormalized weight is always 1, so the task loss
            # would no longer train the router. That variant needs its own gate.
            raise ValueError("top_k must be between 2 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        # Stacked weights keep the door open for a grouped-GEMM backend.
        self.w_gate = nn.Parameter(torch.empty(num_experts, hidden_size, expert_ffn_size))
        self.w_up = nn.Parameter(torch.empty(num_experts, hidden_size, expert_ffn_size))
        self.w_down = nn.Parameter(torch.empty(num_experts, expert_ffn_size, hidden_size))
        self.shared = nn.ModuleList(SwiGLU(hidden_size, expert_ffn_size) for _ in range(num_shared_experts))
        self.aux_records: list[dict] = []
        # Stats of the most recent forward in any mode, for diagnostics such
        # as expert load at inference. Only training mode appends to aux_records.
        self.last_aux: dict | None = None
        # Switched off only while activation checkpointing recomputes this layer.
        self.record_aux = True

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if valid is None:
            index = torch.arange(flat.shape[0], device=x.device)
        else:
            index = valid.reshape(-1).nonzero(as_tuple=True)[0]
        h = flat[index]
        out = torch.zeros_like(flat)
        if h.shape[0] == 0:
            return out.view(shape)

        # Router math must stay in fp32 even inside a bf16 autocast region.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(h.float(), self.router.weight.float())
            probs = logits.softmax(dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        weights = (top_p / top_p.sum(dim=-1, keepdim=True)).to(h.dtype)

        # Sort the (token, expert) assignments by expert so each expert gets one
        # contiguous slice. This needs a single host sync (counts.tolist())
        # instead of one nonzero() per expert.
        flat_expert = top_i.reshape(-1)
        order = flat_expert.argsort(stable=True)
        token_of = order // self.top_k
        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        expert_out = self._run_experts(h[token_of], counts)
        routed = torch.zeros_like(h).index_add_(0, token_of, expert_out * weights.reshape(-1)[order, None])
        for expert in self.shared:
            routed = routed + expert(h)
        out[index] = routed

        load = counts.float() / (self.top_k * h.shape[0])
        balance = self.num_experts * (load * probs.mean(dim=0)).sum()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        self.last_aux = {"balance": balance.detach(), "z": z_loss.detach(), "counts": counts.detach()}
        if self.training and self.record_aux:  # inference never collects them, so they would pile up forever
            self.aux_records.append({"balance": balance, "z": z_loss, "counts": counts.detach()})
        return out.view(shape)

    def _run_experts(self, grouped: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """SwiGLU of each expert on its contiguous slice of `grouped` (rows sorted by expert)."""
        outputs = []
        for e, rows in enumerate(grouped.split(counts.tolist())):
            if rows.shape[0]:
                outputs.append((F.silu(rows @ self.w_gate[e]) * (rows @ self.w_up[e])) @ self.w_down[e])
        return torch.cat(outputs)


def build_ffn(config: NuriConfig, stack: str, layer_number: int) -> nn.Module:
    """FFNFactory. `stack` is "input", "readout" or "core"; layer_number is 1-based."""
    moe = config.moe
    use_moe = moe is not None and (
        (stack == "input" and layer_number in moe.input_layers)
        or (stack == "readout" and layer_number in moe.readout_layers)
        or (stack == "core" and moe.core)
    )
    if not use_moe:
        return SwiGLU(config.hidden_size, config.ffn_size)
    return MoEFFN(config.hidden_size, moe.num_experts, moe.expert_ffn_size, moe.top_k,
                  moe.num_shared_experts)
