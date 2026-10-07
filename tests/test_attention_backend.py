"""Verify optional FlashAttention dispatch, exact masks, caches and compact GQA."""

import copy

import pytest
import torch
import torch.nn.functional as F

from nuri import attention_backend
from nuri.config import tiny_config
from nuri.layers import Attention
from nuri.model import NuriModel


@pytest.fixture
def fake_flash(monkeypatch):
    calls = []

    def fixed(q, k, v, **kwargs):
        calls.append((q.shape[2], k.shape[2], kwargs))
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)

    def varlen(q, k, v, cu_q, cu_k, **kwargs):
        assert cu_q.dtype == cu_k.dtype == torch.int32
        parts = []
        for row in range(len(cu_q) - 1):
            parts.append(
                fixed(
                    q[cu_q[row] : cu_q[row + 1]][None],
                    k[cu_k[row] : cu_k[row + 1]][None],
                    v[cu_k[row] : cu_k[row + 1]][None],
                    **kwargs,
                )[0]
            )
        return torch.cat(parts)

    monkeypatch.setattr(attention_backend, "validate_flash_device", lambda *args: None)
    monkeypatch.setattr(
        attention_backend, "flash_functions", lambda backend: (fixed, varlen)
    )
    return calls


@pytest.mark.parametrize("backend", ["flash2", "flash3"])
@pytest.mark.parametrize("with_padding", [False, True])
def test_varlen_cross_attention_values_and_gradients(fake_flash, backend, with_padding):
    torch.manual_seed(4)
    q = torch.randn(3, 4, 2, 64, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(3, 2, 5, 64, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    reference = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
    mask = (
        torch.tensor([[True, False, True, False, True], [False] * 5, [True] * 5])
        if with_padding
        else None
    )
    actual = attention_backend.flash_attention(q, k, v, mask, backend)
    expected = F.scaled_dot_product_attention(
        *reference,
        attn_mask=mask[:, None, None, :] if mask is not None else None,
        enable_gqa=True,
    )
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    for tensor, ref in zip((q, k, v), reference):
        torch.testing.assert_close(tensor.grad, ref.grad, atol=0.04, rtol=0.04)
    assert all(q_heads == 4 and kv_heads == 2 for q_heads, kv_heads, _ in fake_flash)
    assert all(call[2]["causal"] is False for call in fake_flash)
    assert all(("dropout_p" in call[2]) == (backend == "flash2") for call in fake_flash)


def test_all_empty_keys_have_zero_outputs_and_gradients(fake_flash):
    q = torch.randn(2, 4, 3, 64, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, 5, 64, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = attention_backend.flash_attention(
        q, k, v, torch.zeros(2, 5, dtype=torch.bool), "flash2"
    )
    output.sum().backward()
    assert not output.any()
    assert not fake_flash
    assert all(
        tensor.grad is not None and not tensor.grad.any() for tensor in (q, k, v)
    )


def test_query_masks_keep_exact_sdpa_and_state_dict():
    layer = Attention(32, 4, True, 1e-6, num_kv_heads=2)
    reference = copy.deepcopy(layer)
    layer.attention_backend = "flash2"
    inputs = torch.randn(2, 6, 32)
    document_mask = torch.eye(6, dtype=torch.bool)[None].expand(2, -1, -1)
    torch.testing.assert_close(
        layer(inputs, key_mask=document_mask), reference(inputs, key_mask=document_mask)
    )
    assert layer.used_backends == {"sdpa"}
    assert layer.state_dict().keys() == reference.state_dict().keys()


def test_model_configuration_refuses_cpu_and_preserves_layers():
    model = NuriModel(tiny_config())
    with pytest.raises(ValueError, match="CUDA"):
        model.set_attention_backend("flash2")
    assert all(
        layer.attention_backend == "sdpa"
        for layer in model.modules()
        if isinstance(layer, Attention)
    )
    with pytest.raises(ValueError, match="one of"):
        model.set_attention_backend("unknown")


def test_flash3_rejects_a100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(ValueError, match="Hopper"):
        attention_backend.validate_flash_device("flash3", torch.device("cuda"))


def test_fp32_is_not_silently_cast(fake_flash):
    q = torch.randn(1, 2, 3, 64)
    with pytest.raises(ValueError, match="FP16 or BF16"):
        attention_backend.flash_attention(q, q, q, None, "flash2")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA and optional FlashAttention"
)
@pytest.mark.parametrize("backend", ["flash2", "flash3"])
def test_real_cuda_kernel_matches_sdpa(backend):
    if backend == "flash3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashAttention 3 requires Hopper")
    try:
        attention_backend.flash_functions(backend)
    except RuntimeError as error:
        pytest.skip(str(error))
    layer = Attention(256, 4, True, 1e-6, num_kv_heads=2).cuda()
    reference = copy.deepcopy(layer)
    layer.attention_backend = backend
    inputs = torch.randn(2, 9, 256, device="cuda", requires_grad=True)
    ref_input = inputs.detach().clone().requires_grad_()
    valid = torch.tensor([[True] * 9, [True] * 5 + [False] * 4], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = layer(inputs, key_mask=valid)
        expected = reference(ref_input, key_mask=valid)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(inputs.grad, ref_input.grad, atol=0.002, rtol=0.04)
    for parameter, ref in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref.grad, atol=0.003, rtol=0.04)
