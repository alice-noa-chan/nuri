"""Nuri model: encoder + recurrent core + readout + heads (ARCHITECTURE.md §2-§7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import NuriConfig
from .core import RecurrentReasoningCore
from .encoder import InputEncoder
from .moe import MoEFFN
from .readout import BidirectionalReadout, MemoryCache


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class NuriModel(nn.Module):
    def __init__(self, config: NuriConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.embedding = nn.Embedding(config.vocab_size, d)
        self.encoder = InputEncoder(config)
        self.core = RecurrentReasoningCore(config)
        self.readout = BidirectionalReadout(config)
        # Embedding head (§6): z_fast = P(u), z_reason = P(u + W_r r).
        self.embed_proj = nn.Linear(d, config.embed_dim, bias=False)
        self.reason_proj = nn.Linear(d, d, bias=False)
        # Length classes 1..M including EOS; class index = length - 1.
        self.length_head = nn.Linear(2 * d, config.max_response_len, bias=False)
        self.task_head = nn.Linear(2 * d, config.num_labels, bias=False) if config.num_labels else None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, MoEFFN):
            for p in (module.w_gate, module.w_up, module.w_down):
                nn.init.normal_(p, std=std)

    # ----- input memory and reasoning -------------------------------------------------

    def input_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids != self.config.pad_id

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        valid = self.input_mask(input_ids) if attention_mask is None else attention_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("every sample needs at least one valid input token")
        return self.encoder(self.embedding(input_ids), valid), valid

    def reason(self, x: torch.Tensor, valid: torch.Tensor, steps: int | None = None, return_states: bool = False):
        steps = self.config.default_depth if steps is None else steps
        return self.core(x, valid, steps, return_states=return_states)

    def pool(self, x: torch.Tensor, valid: torch.Tensor, r: torch.Tensor | None = None):
        u = masked_mean(x, valid)
        return u, (r.mean(dim=1) if r is not None else None)

    # ----- heads -------------------------------------------------------------------

    def vocab_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Tied vocabulary projection. Pass only the positions that need logits."""
        return F.linear(hidden, self.embedding.weight)

    def mlm_logits(self, x: torch.Tensor, select: torch.Tensor | None = None) -> torch.Tensor:
        """Token-level MLM head reads X directly (bypasses the core, §6)."""
        return self.vocab_logits(x[select] if select is not None else x)

    def embed(self, input_ids, attention_mask=None, mode: str = "fast", steps: int | None = None,
              dim: int | None = None) -> torch.Tensor:
        x, valid = self.encode(input_ids, attention_mask)
        return self.embed_memory(x, valid, mode, steps, dim)

    def embed_memory(self, x: torch.Tensor, valid: torch.Tensor, mode: str = "fast", steps: int | None = None,
                     dim: int | None = None) -> torch.Tensor:
        """Embedding from an encoded X, so one encoder pass can serve both modes."""
        u, _ = self.pool(x, valid)
        if mode == "reasoning":
            _, r = self.pool(x, valid, self.reason(x, valid, steps))
            u = u + self.reason_proj(r)
        elif mode != "fast":
            raise ValueError(f"unknown embedding mode {mode!r}")
        z = self.embed_proj(u)
        if dim is not None:
            z = z[:, :dim]  # Matryoshka prefix
        return F.normalize(z, dim=-1)

    def length_logits(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.length_head(torch.cat([u, r], dim=-1))

    def task_logits(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.task_head is None:
            raise ValueError("config.num_labels is 0")
        return self.task_head(torch.cat([u, r], dim=-1))

    # ----- generation path -------------------------------------------------------------

    def generation_memory(self, x: torch.Tensor, valid: torch.Tensor, r: torch.Tensor) -> MemoryCache:
        return self.readout.prepare_memory(x, valid, r)

    def readout_hidden(self, canvas_ids: torch.Tensor, canvas_valid: torch.Tensor, cache: MemoryCache):
        return self.readout(self.embedding(canvas_ids), canvas_valid, cache)

    # ----- MoE auxiliary losses ----------------------------------------------------------

    def moe_layers(self) -> list[MoEFFN]:
        return [m for m in self.modules() if isinstance(m, MoEFFN)]

    def pop_moe_aux(self) -> dict:
        """Collect and clear the MoE auxiliary records of every forward since the last call.

        For each layer the records are averaged over its forwards (so running
        a layer twice in one step does not double its weight); layers are
        then summed. Counts are summed per layer for logging.
        """
        moe = self.config.moe
        layers = [m for m in self.moe_layers() if m.aux_records]
        if moe is None or not layers:
            return {"loss": None, "counts": []}
        balance = sum(sum(r["balance"] for r in m.aux_records) / len(m.aux_records) for m in layers)
        z = sum(sum(r["z"] for r in m.aux_records) / len(m.aux_records) for m in layers)
        counts = [sum(r["counts"] for r in m.aux_records) for m in layers]
        for m in layers:
            m.aux_records = []
        loss = moe.balance_loss_weight * balance + moe.z_loss_weight * z
        return {"loss": loss, "balance": balance.detach(), "z": z.detach(), "counts": counts}
