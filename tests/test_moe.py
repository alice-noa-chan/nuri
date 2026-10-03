import pytest
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
    assert moe.aux_records[-1]["counts"].sum().item() == moe.top_k * 8
    x2 = x.clone()
    x2[0, 3:] = 1e3
    assert torch.allclose(moe(x2, valid), out)


def test_aux_losses_reach_router():
    moe = make_moe()
    moe(torch.randn(4, 6, 16))
    (moe.aux_records[-1]["balance"] + moe.aux_records[-1]["z"]).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


def test_factory_places_moe_layers():
    config = tiny_config(input_layers=4, moe=MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2,
                                                       input_layers=(2, 4), readout_layers=(2,)))
    kinds = [type(build_ffn(config, "input", i)) for i in range(1, 5)]
    assert kinds == [SwiGLU, MoEFFN, SwiGLU, MoEFFN]
    assert isinstance(build_ffn(config, "readout", 2), MoEFFN)
    assert isinstance(build_ffn(config, "core", 1), SwiGLU)
    assert isinstance(build_ffn(tiny_config(), "input", 2), SwiGLU)


def test_rejects_top1_routing():
    with pytest.raises(ValueError):
        MoEFFN(hidden_size=16, num_experts=4, expert_ffn_size=8, top_k=1, num_shared_experts=0)


def test_router_stays_fp32_under_autocast():
    moe = make_moe()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        moe(torch.randn(2, 3, 16))
    assert moe.aux_records[-1]["z"].dtype == torch.float32


def test_core_moe_routes_every_step():
    from nuri.model import NuriModel

    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(), readout_layers=(), core=True)
    torch.manual_seed(0)
    model = NuriModel(tiny_config(moe=moe)).train()
    assert isinstance(model.core.ffn, MoEFFN) and len(model.moe_layers()) == 1
    ids = torch.randint(3, 128, (2, 6))
    x, valid = model.encode(ids)
    model.reason(x, valid, steps=3)
    counts = model.core.last_step_expert_counts
    assert len(counts) == 3 and all(int(c.sum()) == 2 * model.config.latent_slots * 2 for c in counts)
    aux = model.pop_moe_aux()
    assert aux["loss"] is not None and int(aux["counts"][0].sum()) == 3 * 2 * model.config.latent_slots * 2
    with torch.no_grad():
        model.eval()
        model.reason(x, valid, steps=5, halt_tol=float("inf"))  # halting works with a MoE core
        assert model.core.last_steps_used.tolist() == [1, 1]


@pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="torch._grouped_mm is not available")
def test_grouped_backend_matches_the_loop_including_gradients():
    loop = make_moe()
    grouped = make_moe()
    grouped.backend = "grouped"
    x = torch.randn(3, 7, 16)
    x_loop, x_grouped = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_loop, out_grouped = loop(x_loop), grouped(x_grouped)
    assert torch.allclose(out_loop, out_grouped, atol=1e-5)
    out_loop.sum().backward()
    out_grouped.sum().backward()  # .sum() gives a zero-stride gradient; the hook must handle it
    assert torch.allclose(x_loop.grad, x_grouped.grad, atol=1e-5)
    for name in ("w_gate", "w_up", "w_down"):
        assert torch.allclose(getattr(loop, name).grad, getattr(grouped, name).grad, atol=1e-5), name


def test_grouped_backend_falls_back_when_unaligned():
    torch.manual_seed(0)
    moe = MoEFFN(hidden_size=6, num_experts=3, expert_ffn_size=5, top_k=2, num_shared_experts=0, backend="grouped")
    for p in (moe.w_gate, moe.w_up, moe.w_down):
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(2, 4, 6)
    assert torch.allclose(moe(x), naive(moe, x), atol=1e-5)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        tiny_config(moe=MoEConfig(backend="fast"))


@pytest.mark.parametrize("device,dtype,capability,expected", [
    ("cuda", torch.bfloat16, (9, 0), True),    # H100
    ("cuda", torch.bfloat16, (8, 0), False),   # A100 must fall back
    ("cuda", torch.float32, (9, 0), False),    # fp32 on CUDA must fall back
    ("cpu", torch.float32, None, True),
    ("cpu", torch.bfloat16, None, False),
])
def test_grouped_support_matrix(device, dtype, capability, expected):
    from nuri.moe import grouped_mm_supported
    assert grouped_mm_supported(device, dtype, capability) is expected


def test_grouped_backend_falls_back_for_unsupported_dtype_and_autocast():
    moe = make_moe()
    moe.backend = "grouped"
    x = torch.randn(2, 5, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):  # CPU bf16 is not a supported grouped case
        out = moe(x)
    assert torch.isfinite(out).all()
    reference = make_moe()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.allclose(out.float(), reference(x).float(), atol=1e-2)
