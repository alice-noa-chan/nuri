"""Adapter contracts on CPU and numerical parity with real Liger when CUDA is available."""

import copy
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from nuri import liger_backend
from nuri.config import tiny_config
from nuri.layers import RMSNorm, SwiGLU
from nuri.losses import liger_token_mean_ce, linear_sample_mean_ce, sample_mean_ce
from nuri.model import NuriModel


@pytest.fixture
def fake_liger(monkeypatch):
    calls = []

    def norm(x, weight, eps, *, offset, casting_mode, in_place):
        calls.append("norm")
        assert offset == 0 and casting_mode == "gemma" and not in_place
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            * weight.float()
        ).to(x.dtype)

    def swiglu(a, b):
        calls.append("swiglu")
        return F.silu(a) * b

    def ce(logits, targets, *, reduction):
        calls.append("ce")
        assert reduction == "none"
        return F.cross_entropy(logits, targets, reduction=reduction)

    def fused(hidden, weight, targets, *, reduction, accum_dtype):
        calls.append("fused")
        assert (
            hidden.requires_grad
            and reduction == "mean"
            and accum_dtype == torch.float32
        )
        return F.cross_entropy(F.linear(hidden, weight).float(), targets)

    monkeypatch.setattr(liger_backend, "validate_liger_device", lambda device: None)
    monkeypatch.setattr(
        liger_backend,
        "liger_functions",
        lambda: {
            "rms_norm": norm,
            "swiglu": swiglu,
            "cross_entropy": ce,
            "fused_linear_cross_entropy": fused,
        },
    )
    return calls


def test_liger_model_preserves_weights_outputs_and_residual_gradients(fake_liger):
    torch.manual_seed(10)
    reference = NuriModel(tiny_config())
    fused = copy.deepcopy(reference)
    fused.set_liger_kernels()
    assert fused.state_dict().keys() == reference.state_dict().keys()
    ids = torch.randint(5, reference.config.vocab_size, (2, 7))
    valid = torch.ones_like(ids, dtype=torch.bool)
    expected = reference.encode(ids, valid)[0]
    actual = fused.encode(ids, valid)[0]
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), fused.named_parameters()
    ):
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, msg=name)
    assert "norm" in fake_liger and "swiglu" in fake_liger
    fused.set_liger_kernels(False)
    assert all(
        not layer.liger_enabled
        for layer in fused.modules()
        if isinstance(layer, (RMSNorm, SwiGLU))
    )


@pytest.mark.parametrize("train_hidden", [True, False])
def test_fused_mean_supports_scaled_gradients_and_frozen_hidden(
    fake_liger, train_hidden
):
    torch.manual_seed(11)
    hidden = torch.randn(5, 8, requires_grad=train_hidden)
    weight = torch.randn(17, 8, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_(train_hidden)
    other_weight = weight.detach().clone().requires_grad_()
    targets = torch.arange(5)
    expected = F.cross_entropy(F.linear(hidden, weight), targets)
    actual = liger_token_mean_ce(other_hidden, other_weight, targets)
    torch.testing.assert_close(actual, expected)
    (expected * 0.37).backward()
    (actual * 0.37).backward()
    torch.testing.assert_close(other_weight.grad, weight.grad)
    if train_hidden:
        torch.testing.assert_close(other_hidden.grad, hidden.grad)


def test_canvas_keeps_unequal_row_weights_and_avoids_fused_none(fake_liger):
    torch.manual_seed(12)
    mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False], [False] * 4]
    )
    targets = torch.randint(0, 17, mask.shape)
    hidden = torch.randn(5, 8, requires_grad=True)
    weight = torch.randn(17, 8, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_()
    other_weight = weight.detach().clone().requires_grad_()
    expected = sample_mean_ce(F.linear(hidden, weight), targets, mask)
    actual = linear_sample_mean_ce(
        other_hidden, other_weight, targets, mask, 2, ce_backend="liger"
    )
    torch.testing.assert_close(actual, expected)
    (expected * 0.37).backward()
    (actual * 0.37).backward()
    torch.testing.assert_close(other_hidden.grad, hidden.grad)
    torch.testing.assert_close(other_weight.grad, weight.grad)
    assert fake_liger and set(fake_liger) == {"ce"}


def test_cpu_enable_is_rejected_without_changing_flags():
    model = NuriModel(tiny_config())
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        model.set_liger_kernels()
    assert all(
        not layer.liger_enabled
        for layer in model.modules()
        if isinstance(layer, (RMSNorm, SwiGLU))
    )


def real_liger_available():
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("liger_kernel") is not None
    )


@pytest.mark.skipif(not real_liger_available(), reason="requires CUDA and Liger")
@pytest.mark.parametrize("canvas", [False, True])
def test_real_liger_loss_and_scaled_gradients(canvas):
    torch.manual_seed(13)
    hidden = torch.randn(5, 64, device="cuda", requires_grad=True)
    weight = torch.randn(127, 64, device="cuda", requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_()
    other_weight = weight.detach().clone().requires_grad_()
    targets = torch.randint(0, 127, (2, 4), device="cuda")
    mask = torch.tensor([[True] * 4, [True, False, False, False]], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = F.linear(hidden, weight).float()
        expected = (
            sample_mean_ce(logits, targets, mask)
            if canvas
            else F.cross_entropy(logits, targets[mask])
        )
        actual = (
            linear_sample_mean_ce(
                other_hidden, other_weight, targets, mask, 2, ce_backend="liger"
            )
            if canvas
            else liger_token_mean_ce(other_hidden, other_weight, targets[mask])
        )
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.01)
    (expected * 0.37).backward()
    (actual * 0.37).backward()
    torch.testing.assert_close(other_hidden.grad, hidden.grad, atol=0.02, rtol=0.05)
    torch.testing.assert_close(other_weight.grad, weight.grad, atol=0.02, rtol=0.05)


@pytest.mark.skipif(not real_liger_available(), reason="requires CUDA and Liger")
def test_real_liger_residual_norm_and_swiglu():
    torch.manual_seed(14)
    reference = torch.nn.Sequential(RMSNorm(64), SwiGLU(64, 128)).cuda()
    fused = copy.deepcopy(reference)
    for layer in fused:
        layer.liger_enabled = True
    # Transposition exercises contiguous adapters; the residual shares upstream gradients.
    x = (
        torch.randn(2, 64, 7, device="cuda", dtype=torch.bfloat16)
        .transpose(1, 2)
        .requires_grad_()
    )
    other_x = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(x) + x
        actual = fused(other_x) + other_x
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(other_x.grad, x.grad, atol=0.002, rtol=0.05)
    for a, b in zip(reference.parameters(), fused.parameters()):
        torch.testing.assert_close(b.grad, a.grad, atol=0.002, rtol=0.05)
