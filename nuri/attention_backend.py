"""Optional CUDA attention kernels; model weights and masks keep their meaning."""

from functools import lru_cache
from importlib import import_module

import torch

ATTENTION_BACKENDS = ("sdpa", "flash2", "flash3")


@lru_cache(maxsize=2)
def flash_functions(backend: str):
    """Import kernels only when requested, including the legacy FA3 package layout."""
    try:
        if backend == "flash2":
            module = import_module("flash_attn")
        elif backend == "flash3":
            try:
                module = import_module("flash_attn_3.flash_attn_interface")
            except ModuleNotFoundError as error:
                if not error.name or not error.name.startswith("flash_attn_3"):
                    raise
                module = import_module("flash_attn_interface")
        else:
            raise ValueError(f"unknown FlashAttention backend: {backend!r}")
        return module.flash_attn_func, module.flash_attn_varlen_func
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"{backend} needs its CUDA FlashAttention package, including fixed and varlen APIs. "
            "Install the matching upstream package on the GPU host or select sdpa."
        ) from error


def validate_flash_device(backend: str, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(f"{backend} requires an NVIDIA CUDA device")
    major, _ = torch.cuda.get_device_capability(device)
    if torch.version.hip is not None or major < 8:
        raise ValueError(f"{backend} requires an NVIDIA Ampere or newer GPU")
    if backend == "flash3" and major != 9:
        raise ValueError("flash3 requires a Hopper GPU; use flash2 on A100")


def validate_attention_backend(
    backend: str, device: torch.device, head_dims=()
) -> None:
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"attention backend must be one of {ATTENTION_BACKENDS}")
    if backend == "sdpa":
        return
    validate_flash_device(backend, device)
    if any(dim > 256 or dim % 8 for dim in head_dims):
        raise ValueError(
            "FlashAttention requires head dimensions divisible by 8 and at most 256"
        )
    if backend == "flash3" and any(dim not in (64, 128, 256) for dim in head_dims):
        raise ValueError(
            "flash3 supports head dimensions 64, 128 and 256 in this adapter"
        )
    flash_functions(backend)


def _output_tensor(output):
    # Older Hopper interfaces return output plus diagnostic tensors.
    return output[0] if isinstance(output, tuple) else output


def flash_attention(q, k, v, key_mask, backend: str):
    """Run exact non-causal attention with compact GQA heads and optional key padding.

    All query positions are retained, preserving SDPA's padded-query outputs.
    Only masked keys/values are removed. Query-specific masks stay on the SDPA
    path in Attention; they must never be discarded to force a fused kernel.
    """
    validate_flash_device(backend, q.device)
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"{backend} requires FP16 or BF16 queries; enable CUDA BF16 autocast"
        )
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.device != q.device
        or v.device != q.device
    ):
        raise ValueError(
            "FlashAttention queries, keys and values need matching devices and dtypes"
        )
    fixed, varlen = flash_functions(backend)
    arguments = {
        "causal": False,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    if backend == "flash2":
        arguments["dropout_p"] = 0.0
    queries, keys, values = (
        tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)
    )
    if key_mask is None:
        return _output_tensor(fixed(queries, keys, values, **arguments)).transpose(1, 2)
    if key_mask.dtype != torch.bool or key_mask.shape != keys.shape[:2]:
        raise ValueError(
            "FlashAttention key padding must be boolean with shape [batch, keys]"
        )

    lengths = key_mask.sum(dim=1, dtype=torch.int32)
    active = (lengths > 0).nonzero(as_tuple=True)[0]
    if not active.numel():
        return q * 0 + (k.sum() + v.sum()) * 0
    batch, query_length = active.numel(), queries.shape[1]
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * query_length
    cu_k = torch.cat(
        [lengths.new_zeros(1), lengths[active].cumsum(0, dtype=torch.int32)]
    )
    packed_q = queries[active].reshape(-1, q.shape[1], q.shape[-1])
    packed_k, packed_v = keys[key_mask], values[key_mask]
    output = _output_tensor(
        varlen(
            packed_q,
            packed_k,
            packed_v,
            cu_q,
            cu_k,
            max_seqlen_q=query_length,
            max_seqlen_k=keys.shape[1],
            **arguments,
        )
    ).reshape(batch, query_length, q.shape[1], q.shape[-1])
    output = (queries * 0).index_copy(0, active, output)
    return output.transpose(1, 2)
