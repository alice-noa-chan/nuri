"""Shared building blocks: RMSNorm, RoPE, attention, SwiGLU (ARCHITECTURE.md §4, §9)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention_backend import flash_attention, validate_attention_backend


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
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(
        torch.arange(length, device=device, dtype=torch.float32), inv_freq
    )
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(
    x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
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

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool,
        eps: float,
        num_kv_heads: int | None = None,
        kv_latent_dim: int | None = None,
    ):
        """num_kv_heads < num_heads gives grouped-query attention (GQA): several
        query heads share one key/value head, so K/V projections and cached
        K/V shrink by num_heads / num_kv_heads. None means plain MHA.

        kv_latent_dim (MLA-style, experimental) first compresses the memory to
        a small shared latent, then expands it to K and V:
        K = W_k_up(W_down(x)), V = W_v_up(W_down(x)). This tests the
        projection bottleneck itself (§9); None means direct K/V projections.
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.native_gqa = False
        self.attention_backend = "sdpa"
        self.used_backends: set[str] = set()
        self.head_dim = hidden_size // num_heads
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        kv_input = kv_latent_dim or hidden_size
        self.kv_down = (
            nn.Linear(hidden_size, kv_latent_dim, bias=False) if kv_latent_dim else None
        )
        self.k_proj = nn.Linear(kv_input, kv_size, bias=False)
        self.v_proj = nn.Linear(kv_input, kv_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps) if qk_norm else None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, size = x.shape
        return x.view(b, n, size // self.head_dim, self.head_dim).transpose(1, 2)

    def set_native_gqa(self, enabled: bool = True) -> None:
        """Select native SDPA GQA at runtime; the portable path remains default."""
        configure_native_gqa(self, enabled)

    def project_kv(
        self, memory: torch.Tensor, rope=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reusable K/V with num_kv_heads heads for either attention path."""
        if self.kv_down is not None:
            memory = self.kv_down(memory)  # shared low-dimensional latent for K and V
        k = self._split(self.k_proj(memory))
        if self.k_norm is not None:
            k = self.k_norm(k)
        if rope is not None:
            k = apply_rope(k, rope)
        return k, self._split(self.v_proj(memory))

    def forward(self, x, kv=None, key_mask=None, rope=None) -> torch.Tensor:
        """x: queries [B, n, D]. kv: cached (k, v) or None for self-attention.

        key_mask: bool [B, n_keys] (same keys for every query) or
            [B, n, n_keys] (per query, e.g. local attention); True = may attend.
        rope: tables applied to queries (and keys when kv is None).
        """
        q = self._split(self.q_proj(x))
        if self.q_norm is not None:
            q = self.q_norm(q)
        if rope is not None:
            q = apply_rope(q, rope)
        k, v = kv if kv is not None else self.project_kv(x, rope)
        # Document isolation and local/group masks must retain every allowed edge.
        # The optional varlen kernels handle key padding, not arbitrary query masks.
        use_flash = self.attention_backend != "sdpa" and (
            key_mask is None or key_mask.dim() == 2
        )
        if use_flash:
            out = flash_attention(q, k, v, key_mask, self.attention_backend)
            self.used_backends.add(self.attention_backend)
            b, _, n, _ = out.shape
            return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))
        self.used_backends.add("sdpa")
        grouped = self.num_kv_heads != self.num_heads
        if grouped and not self.native_gqa:
            group = self.num_heads // self.num_kv_heads
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        if key_mask is None:
            attn_mask = None
        elif key_mask.dim() == 2:
            attn_mask = key_mask[:, None, None, :]
        else:
            attn_mask = key_mask[:, None, :, :]
        if grouped and self.native_gqa:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        b, _, n, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


def configure_attention_backend(module: nn.Module, backend: str = "sdpa") -> None:
    """Select a runtime kernel atomically; parameter names and shapes stay unchanged."""
    layers = [layer for layer in module.modules() if isinstance(layer, Attention)]
    device = next(module.parameters()).device
    validate_attention_backend(backend, device, [layer.head_dim for layer in layers])
    for layer in layers:
        layer.attention_backend = backend
        layer.used_backends.clear()


def configure_native_gqa(module: nn.Module, enabled: bool = True) -> None:
    """Configure all Attention descendants without changing checkpoint weights.

    A deterministic CPU math probe checks the native API before any flag is
    changed. Actual backend selection still depends on the device, dtype and
    mask; forced incompatible backends report their normal SDPA error. Native
    support alone does not guarantee a fused kernel or a speed improvement.
    """
    attention_layers = [
        layer for layer in module.modules() if isinstance(layer, Attention)
    ]
    if enabled and any(
        layer.num_kv_heads != layer.num_heads for layer in attention_layers
    ):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        query = torch.zeros(1, 2, 1, 2, device="cpu", dtype=torch.float32)
        memory = torch.zeros(1, 1, 1, 2, device="cpu", dtype=torch.float32)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                F.scaled_dot_product_attention(query, memory, memory, enable_gqa=True)
        except (TypeError, RuntimeError) as error:
            raise RuntimeError(
                "This PyTorch build does not support SDPA enable_gqa. Upgrade to "
                "PyTorch >=2.5 with native GQA support, or disable native GQA to "
                "use the portable repeated-head path."
            ) from error
    for layer in attention_layers:
        layer.native_gqa = bool(enabled)


def run_checkpointed(module: nn.Module, *args):
    """Run `module(*args)` without keeping its activations; they are recomputed in backward.

    The recomputation runs the module a second time. MoE layers inside must
    not record their auxiliary losses again, or a step would count them
    twice, so recording is switched off for every call after the first.
    """
    moe_layers = [m for m in module.modules() if hasattr(m, "record_aux")]
    calls = []

    def forward(*inputs):
        calls.append(1)
        for layer in moe_layers:
            layer.record_aux = len(calls) == 1
        try:
            return module(*inputs)
        finally:
            for layer in moe_layers:
                layer.record_aux = True

    return checkpoint(forward, *args, use_reentrant=False)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        # `valid` is accepted so dense and MoE FFNs share one call signature.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
