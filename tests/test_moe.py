import torch
import torch.nn.functional as F

from nuri.config import MoEConfig, tiny_config
from nuri.layers import SwiGLU
from nuri.moe import MoEFFN, build_ffn


def make_moe(seed=0):
    torch.manual_seed(seed)
    moe = MoEFFN(hidden_size=16, num_experts=6, expert_ffn_size=8, top_k=2, num_shared_experts=1)
    for p in (moe.w_gate, moe.w_up, moe.w_down):
        torch.nn.init.normal_(p, std=0.2)
    return moe


def naive(moe, x):
    """Per-token reference: shared + renormalized top-k experts."""
    outs = []
    for h in x.reshape(-1, x.shape[-1]):
        probs = F.linear(h.float(), moe.router.weight.float()).softmax(-1)
        top_p, top_i = probs.topk(moe.top_k)
        top_p = top_p / top_p.sum()
        y = sum(s(h) for s in moe.shared)
        for p, e in zip(top_p, top_i):
            y = y + p * ((F.silu(h @ moe.w_gate[e]) * (h @ moe.w_up[e])) @ moe.w_down[e])
        outs.append(y)
    return torch.stack(outs).view(x.shape)


def test_matches_naive_routing():
    moe = make_moe()
    x = torch.randn(2, 5, 16)
    assert torch.allclose(moe(x), naive(moe, x), atol=1e-5)


def test_padding_not_routed_or_counted():
    moe = make_moe()
    x = torch.randn(2, 5, 16)
    valid = torch.tensor([[True] * 3 + [False] * 2, [True] * 5])
    out = moe(x, valid)
    assert torch.all(out[0, 3:] == 0)
    assert moe.last_aux["counts"].sum().item() == moe.top_k * 8
    x2 = x.clone()
    x2[0, 3:] = 1e3
    assert torch.allclose(moe(x2, valid), out)


def test_aux_losses_reach_router():
    moe = make_moe()
    moe(torch.randn(4, 6, 16))
    (moe.last_aux["balance"] + moe.last_aux["z"]).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


def test_factory_places_moe_layers():
    config = tiny_config(input_layers=4, moe=MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2,
                                                       input_layers=(2, 4), readout_layers=(2,)))
    kinds = [type(build_ffn(config, "input", i)) for i in range(1, 5)]
    assert kinds == [SwiGLU, MoEFFN, SwiGLU, MoEFFN]
    assert isinstance(build_ffn(config, "readout", 2), MoEFFN)
    assert isinstance(build_ffn(config, "core", 1), SwiGLU)
    assert isinstance(build_ffn(tiny_config(), "input", 2), SwiGLU)
