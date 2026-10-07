"""Optional Liger functional kernels; model parameters remain ordinary PyTorch tensors."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def liger_functions():
    try:
        functional = importlib.import_module("liger_kernel.transformers.functional")
        return {
            name: getattr(functional, "liger_" + name)
            for name in (
                "rms_norm",
                "swiglu",
                "cross_entropy",
                "fused_linear_cross_entropy",
            )
        }
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Liger requires liger-kernel>=0.6.4 and a compatible Triton/CUDA installation."
        ) from error


def validate_liger_device(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip:
        raise ValueError("Liger kernels require an NVIDIA CUDA device")


def validate_liger(device: torch.device) -> None:
    validate_liger_device(device)
    liger_functions()


def rms_norm(x, weight, eps):
    validate_liger_device(x.device)
    # Gemma casting matches Nuri's FP32 normalization and weight multiplication.
    # Out-of-place backward protects gradients shared with residual branches.
    return liger_functions()["rms_norm"](
        x.contiguous(), weight, eps, offset=0.0, casting_mode="gemma", in_place=False
    )


def swiglu(gate, up):
    validate_liger_device(gate.device)
    return liger_functions()["swiglu"](gate.contiguous(), up.contiguous())


def cross_entropy(logits, targets):
    validate_liger_device(logits.device)
    return liger_functions()["cross_entropy"](logits, targets, reduction="none")


def fused_linear_ce(hidden, weight, targets):
    validate_liger_device(hidden.device)
    # Older supported Liger versions compute weight gradients only when the
    # input requires gradients. A detached leaf also supports a frozen encoder.
    if torch.is_grad_enabled() and weight.requires_grad and not hidden.requires_grad:
        hidden = hidden.detach().requires_grad_()
    return liger_functions()["fused_linear_cross_entropy"](
        hidden.contiguous(),
        weight,
        targets,
        reduction="mean",
        accum_dtype=torch.float32,
    )
