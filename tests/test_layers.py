import math

import torch

from nuri.layers import Attention, RMSNorm, SwiGLU, apply_rope, rope_cache


def test_rmsnorm_unit_rms():
    norm = RMSNorm(16)
    y = norm(torch.randn(3, 5, 16) * 7)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)


def test_rope_is_relative():
    """q·k after RoPE depends only on the position offset."""
    torch.manual_seed(0)
    rope = rope_cache(16, 8, 10000.0, "cpu")
    q, k = torch.randn(8), torch.randn(8)

    def score(i, j):
        seq_q = torch.zeros(1, 1, 16, 8)
        seq_k = torch.zeros(1, 1, 16, 8)
        seq_q[0, 0, i], seq_k[0, 0, j] = q, k
        return (apply_rope(seq_q, rope)[0, 0, i] * apply_rope(seq_k, rope)[0, 0, j]).sum()

    assert torch.allclose(score(3, 1), score(10, 8), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)


def test_attention_ignores_masked_keys():
    torch.manual_seed(0)
    attn = Attention(32, 4, qk_norm=True, eps=1e-6)
    x = torch.randn(2, 6, 32)
    mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    out = attn(x, key_mask=mask)
    x2 = x.clone()
    x2[0, 4:] = torch.randn(2, 32) * 100
    out2 = attn(x2, key_mask=mask)
    assert torch.allclose(out[0, :4], out2[0, :4], atol=1e-5)


def test_cached_kv_matches_inline():
    torch.manual_seed(0)
    attn = Attention(32, 4, qk_norm=True, eps=1e-6)
    q, mem = torch.randn(2, 3, 32), torch.randn(2, 7, 32)
    kv = attn.project_kv(mem)
    manual = attn(q, kv=kv)
    # Reference computation without SDPA.
    qh = attn.q_norm(attn._split(attn.q_proj(q)))
    k, v = kv
    weights = torch.softmax(qh @ k.transpose(-1, -2) / math.sqrt(attn.head_dim), -1)
    ref = attn.o_proj((weights @ v).transpose(1, 2).reshape(2, 3, 32))
    assert torch.allclose(manual, ref, atol=1e-5)


def test_swiglu_shape():
    assert SwiGLU(16, 40)(torch.randn(2, 3, 16)).shape == (2, 3, 16)


def test_gqa_shares_key_value_heads():
    torch.manual_seed(0)
    gqa = Attention(32, 4, qk_norm=True, eps=1e-6, num_kv_heads=2)
    assert gqa.k_proj.weight.shape == (16, 32)  # 2 heads x head_dim 8
    x = torch.randn(2, 5, 32)
    k, v = gqa.project_kv(x)
    assert k.shape == (2, 2, 5, 8)  # the cache keeps 2 heads, not 4
    # Same as MHA whose 4 K/V heads repeat the 2 shared ones.
    mha = Attention(32, 4, qk_norm=True, eps=1e-6)
    mha.load_state_dict({**gqa.state_dict(),
                         "k_proj.weight": gqa.k_proj.weight.view(2, 8, 32).repeat_interleave(2, 0).reshape(32, 32),
                         "v_proj.weight": gqa.v_proj.weight.view(2, 8, 32).repeat_interleave(2, 0).reshape(32, 32)})
    assert torch.allclose(gqa(x), mha(x), atol=1e-6)
