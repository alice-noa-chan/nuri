"""Native grouped attention preserves portable outputs, caches and gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from nuri.layers import Attention, configure_native_gqa, rope_cache


def _require_native_api():
    query = torch.zeros(1, 2, 1, 2)
    memory = torch.zeros(1, 1, 1, 2)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            F.scaled_dot_product_attention(query, memory, memory, enable_gqa=True)
    except (TypeError, RuntimeError):
        pytest.skip("this PyTorch build does not provide native SDPA GQA")


def _mask(kind, query_length, key_length, device="cpu"):
    if kind == "none":
        return None
    mask = torch.ones(2, key_length, dtype=torch.bool, device=device)
    mask[0, -2:] = False
    if kind == "keys":
        return mask
    mask = mask[:, None, :].expand(-1, query_length, -1).clone()
    mask[:, 1::2, 0] = False
    return mask


def _assert_parameter_gradients_match(portable, native):
    left = dict(portable.named_parameters())
    right = dict(native.named_parameters())
    assert left.keys() == right.keys()
    for name, parameter in left.items():
        assert parameter.grad is not None, name
        assert right[name].grad is not None, name
        torch.testing.assert_close(parameter.grad, right[name].grad, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("mask_kind", ["none", "keys", "per_query"])
@pytest.mark.parametrize("qk_norm,use_rope", [(False, False), (True, True)])
def test_native_self_attention_matches_complete_forward_and_backward(mask_kind, qk_norm, use_rope):
    _require_native_api()
    torch.manual_seed(31)
    portable = Attention(32, 4, qk_norm, 1e-6, num_kv_heads=2).double()
    native = copy.deepcopy(portable)
    native.set_native_gqa()
    inputs = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
    native_inputs = inputs.detach().clone().requires_grad_()
    mask = _mask(mask_kind, 5, 5)
    rope = rope_cache(5, 8, 10000.0, "cpu") if use_rope else None

    with sdpa_kernel(SDPBackend.MATH):
        expected = portable(inputs, key_mask=mask, rope=rope)
        actual = native(native_inputs, key_mask=mask, rope=rope)
    torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-7)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()

    torch.testing.assert_close(inputs.grad, native_inputs.grad, rtol=2e-6, atol=2e-7)
    _assert_parameter_gradients_match(portable, native)


@pytest.mark.parametrize("mask_kind", ["none", "keys", "per_query"])
@pytest.mark.parametrize("qk_norm,use_rope", [(False, False), (True, True)])
@pytest.mark.parametrize("kv_latent_dim", [None, 12])
def test_native_repeated_cross_reads_preserve_memory_and_cache_gradients(
    mask_kind, qk_norm, use_rope, kv_latent_dim
):
    _require_native_api()
    torch.manual_seed(32)
    portable = Attention(32, 4, qk_norm, 1e-6, num_kv_heads=2, kv_latent_dim=kv_latent_dim).double()
    native = copy.deepcopy(portable)
    native.set_native_gqa()
    query = torch.randn(2, 4, 32, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(2, 6, 32, dtype=torch.float64, requires_grad=True)
    native_query = query.detach().clone().requires_grad_()
    native_memory = memory.detach().clone().requires_grad_()
    rope = rope_cache(6, 8, 10000.0, "cpu") if use_rope else None
    mask = _mask(mask_kind, 4, 6)
    portable_kv = portable.project_kv(memory, rope)
    native_kv = native.project_kv(native_memory, rope)
    for cache in (*portable_kv, *native_kv):
        cache.retain_grad()
        assert cache.shape == (2, 2, 6, 8)

    with sdpa_kernel(SDPBackend.MATH):
        expected = portable(query, kv=portable_kv, key_mask=mask, rope=rope)
        expected = expected + portable(query * 1.5, kv=portable_kv, key_mask=mask, rope=rope)
        actual = native(native_query, kv=native_kv, key_mask=mask, rope=rope)
        actual = actual + native(native_query * 1.5, kv=native_kv, key_mask=mask, rope=rope)
    torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-7)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()

    torch.testing.assert_close(query.grad, native_query.grad, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(memory.grad, native_memory.grad, rtol=2e-6, atol=2e-7)
    for expected_cache, actual_cache in zip(portable_kv, native_kv):
        torch.testing.assert_close(expected_cache.grad, actual_cache.grad, rtol=2e-6, atol=2e-7)
    _assert_parameter_gradients_match(portable, native)


def test_native_passes_compact_cached_heads_to_sdpa(monkeypatch):
    _require_native_api()
    attention = Attention(32, 4, True, 1e-6, num_kv_heads=2)
    attention.set_native_gqa()
    query = torch.randn(2, 4, 32)
    cache = attention.project_kv(torch.randn(2, 6, 32))
    original = F.scaled_dot_product_attention
    calls = []

    def capture(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", capture)
    attention(query, kv=cache)
    assert calls[0][0].shape[1] == 4
    assert calls[0][1] is cache[0] and calls[0][2] is cache[1]
    assert calls[0][3]["enable_gqa"] is True
    attention.set_native_gqa(False)
    attention(query, kv=cache)
    assert calls[1][1].shape[1] == calls[1][2].shape[1] == 4
    assert not calls[1][3].get("enable_gqa", False)


def test_configuration_preserves_random_state_and_checkpoint_keys():
    _require_native_api()
    model = nn.ModuleList([Attention(32, 4, False, 1e-6), Attention(32, 4, True, 1e-6, 2)])
    state = copy.deepcopy(model.state_dict())
    random_state = torch.get_rng_state().clone()
    assert all(not attention.native_gqa for attention in model)
    configure_native_gqa(model)
    assert all(attention.native_gqa for attention in model)
    assert torch.equal(random_state, torch.get_rng_state())
    for name, tensor in model.state_dict().items():
        assert torch.equal(state[name], tensor)
    configure_native_gqa(model, False)
    assert all(not attention.native_gqa for attention in model)


def test_unsupported_native_api_fails_before_any_module_is_changed(monkeypatch):
    model = nn.ModuleList([Attention(32, 4, False, 1e-6), Attention(32, 4, True, 1e-6, 2)])

    def old_sdpa(q, k, v, attn_mask=None):
        raise AssertionError("the portable body should not run during the native API probe")

    monkeypatch.setattr(F, "scaled_dot_product_attention", old_sdpa)
    with pytest.raises(RuntimeError, match=r"enable_gqa.*[Uu]pgrade|[Uu]pgrade.*enable_gqa"):
        configure_native_gqa(model)
    assert all(not attention.native_gqa for attention in model)
    configure_native_gqa(model, False)  # disabling never needs native API support


def test_native_backend_errors_are_not_silently_retried(monkeypatch):
    _require_native_api()
    attention = Attention(32, 4, True, 1e-6, num_kv_heads=2)
    attention.set_native_gqa()
    calls = []

    def unavailable_backend(q, k, v, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("the selected backend rejected this native GQA call")

    monkeypatch.setattr(F, "scaled_dot_product_attention", unavailable_backend)
    with pytest.raises(RuntimeError, match="selected backend rejected"):
        attention(torch.randn(2, 5, 32))
    assert len(calls) == 1 and calls[0]["enable_gqa"] is True


def test_mha_remains_compatible_with_the_old_sdpa_api(monkeypatch):
    torch.manual_seed(33)
    attention = Attention(32, 4, True, 1e-6)
    inputs = torch.randn(2, 5, 32)
    expected = attention(inputs)
    original = F.scaled_dot_product_attention

    def old_sdpa(q, k, v, attn_mask=None):
        return original(q, k, v, attn_mask=attn_mask)

    monkeypatch.setattr(F, "scaled_dot_product_attention", old_sdpa)
    attention.set_native_gqa()
    torch.testing.assert_close(attention(inputs), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_bf16_matches_portable_attention():
    _require_native_api()
    torch.manual_seed(34)
    portable = Attention(64, 4, True, 1e-6, num_kv_heads=2).cuda().to(torch.bfloat16)
    native = copy.deepcopy(portable)
    native.set_native_gqa()
    inputs = torch.randn(2, 7, 64, device="cuda", dtype=torch.bfloat16)
    mask = _mask("keys", 7, 7, "cuda")
    rope = rope_cache(7, 16, 10000.0, "cuda")
    torch.testing.assert_close(native(inputs, key_mask=mask, rope=rope),
                               portable(inputs, key_mask=mask, rope=rope), rtol=2e-2, atol=2e-2)
