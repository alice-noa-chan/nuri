"""Independent regressions for experimental core state and gradient contracts."""

import copy

import pytest
import torch

from nuri.config import MoEConfig, tiny_config
from nuri.core import RecurrentReasoningCore
from nuri.model import NuriModel


def core_moe_model(**overrides):
    torch.manual_seed(31)
    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2,
                    input_layers=(), readout_layers=(), core=True)
    return NuriModel(tiny_config(moe=moe, **overrides))


@pytest.mark.parametrize("tolerance,groups", [(float("nan"), 1), (-0.1, 1), (0.0, 2)])
def test_invalid_halting_preserves_pending_state_and_auxiliary_records(tolerance, groups):
    model = core_moe_model(slot_groups=groups).train()
    core = model.core
    x = torch.randn(2, 5, model.config.hidden_size)
    valid = torch.ones(2, 5, dtype=torch.bool)
    core(x, valid, steps=2)
    used = core.last_steps_used
    counts = core.last_step_expert_counts
    records = tuple(core.ffn.aux_records)
    projections = []
    original = core.cross_attn.project_kv
    core.cross_attn.project_kv = lambda *a, **k: projections.append(1) or original(*a, **k)

    with pytest.raises(ValueError, match="halt_tol"):
        core(x, valid, steps=4, halt_tol=tolerance)

    assert projections == []
    assert core.last_steps_used is used
    assert core.last_step_expert_counts is counts
    assert len(core.ffn.aux_records) == len(records)
    assert all(saved is current for saved, current in zip(records, core.ffn.aux_records))


def test_half_precision_halting_does_not_mistake_norm_overflow_for_convergence(monkeypatch):
    core = RecurrentReasoningCore(tiny_config()).half().eval()
    with torch.no_grad():
        core.slots.fill_(13000)
    # Every entry and update is finite in FP16, but the norm of this many
    # entries overflows FP16. The true update ratio stays above 0.05.
    monkeypatch.setattr(core, "_step", lambda r, *a, **k: r + 1024)
    x = torch.zeros(2, 3, core.slots.shape[1], dtype=torch.float16)
    valid = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        out = core(x, valid, steps=3, halt_tol=0.05)
    assert core.last_steps_used.tolist() == [3, 3]
    assert torch.isfinite(out).all()


def test_active_rows_keep_original_memory_and_masks_after_other_rows_halt(monkeypatch):
    core = RecurrentReasoningCore(tiny_config()).eval()
    with torch.no_grad():
        core.slots.fill_(0.1)
    x = torch.zeros(3, 4, core.slots.shape[1])
    x[:, 0, 0] = torch.tensor([1, 2, 3])
    valid = torch.arange(4)[None, :] < torch.tensor([2, 3, 4])[:, None]
    projections, batches = [], []
    row_steps = {1: 0, 2: 0, 3: 0}

    def project(memory):
        projections.append(1)
        key = memory[:, :1, :1].unsqueeze(1)
        return key, key + 10

    def step(state, kv, mask, gates, grouped=False):
        rows = kv[0][:, 0, 0, 0].long().tolist()
        batches.append(rows)
        assert torch.equal(kv[1], kv[0] + 10)
        for row, original_id in enumerate(rows):
            assert torch.equal(mask[row], valid[original_id - 1])
            row_steps[original_id] += 1
        update = torch.zeros_like(state)
        for row, original_id in enumerate(rows):
            terminal = original_id < 3 and row_steps[original_id] >= original_id
            update[row, :, 0] = 0.0001 if terminal else 0.2
        return state + update

    monkeypatch.setattr(core.cross_attn, "project_kv", project)
    monkeypatch.setattr(core, "_step", step)
    with torch.no_grad():
        out = core(x, valid, steps=4, halt_tol=0.01)
        expected = core.slots.unsqueeze(0).expand(3, -1, -1).clone()
        expected[:, :, 0] += torch.tensor([0.0001, 0.2001, 0.8])[:, None]
        expected = core.final_norm(expected)
    assert projections == [1]
    assert batches == [[1, 2, 3], [2, 3], [3], [3]]
    assert core.last_steps_used.tolist() == [1, 2, 4]
    torch.testing.assert_close(out, expected)


def test_zero_tolerance_preserves_core_moe_gradients_and_projects_memory_once():
    fixed = core_moe_model().train()
    adaptive = copy.deepcopy(fixed)
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 0, 0]])
    weights = torch.randn(2, fixed.config.latent_slots, fixed.config.hidden_size)
    gradients, outputs = [], []
    for model, tolerance in ((fixed, None), (adaptive, 0.0)):
        projections = []
        original = model.core.cross_attn.project_kv
        model.core.cross_attn.project_kv = lambda *a, **k: projections.append(1) or original(*a, **k)
        x, valid = model.encode(ids)
        out = model.reason(x, valid, steps=3, halt_tol=tolerance)
        assert projections == [1]
        assert len(model.core.ffn.aux_records) == 3
        aux = model.pop_moe_aux()
        assert int(aux["counts"][0].sum()) == 3 * 2 * model.config.latent_slots * 2
        ((out * weights).sum() + aux["loss"]).backward()
        assert model.core.ffn.aux_records == []
        outputs.append(out.detach())
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(outputs[0], outputs[1])
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-5, atol=2e-6)
    for name in ("embedding.weight", "core.cross_attn.k_proj.weight", "core.gate_logits",
                 "core.ffn.router.weight", "core.ffn.w_gate", "core.ffn.shared.0.down_proj.weight"):
        assert gradients[0][name].abs().sum() > 0


def test_core_moe_task_loss_trains_router_without_auxiliary_loss():
    model = core_moe_model().train()
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    x, valid = model.encode(ids)
    out = model.reason(x, valid, steps=3)
    (out * torch.randn_like(out)).sum().backward()
    for parameter in (model.core.ffn.router.weight, model.core.ffn.w_gate,
                      model.core.ffn.w_up, model.core.ffn.w_down):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_core_moe_step_stats_are_snapshots_and_eval_preserves_pending_training_records():
    model = core_moe_model().train()
    x = torch.randn(2, 5, model.config.hidden_size)
    valid = torch.ones(2, 5, dtype=torch.bool)
    model.reason(x, valid, steps=3)
    snapshots = list(model.core.last_step_expert_counts)
    saved = [counts.clone() for counts in snapshots]
    records = tuple(model.core.ffn.aux_records)
    assert len({counts.data_ptr() for counts in snapshots}) == 3
    assert all(not counts.requires_grad for counts in snapshots)
    with torch.no_grad():
        model.eval()
        model.reason(x[:1], valid[:1], steps=5, halt_tol=float("inf"))
        assert model.core.last_steps_used.tolist() == [1]
        assert len(model.core.last_step_expert_counts) == 1
        assert int(model.core.last_step_expert_counts[0].sum()) == model.config.latent_slots * 2
        model.reason(x, valid, steps=0)
        assert model.core.last_steps_used.tolist() == [0, 0]
        assert model.core.last_step_expert_counts == []
    assert all(torch.equal(old, now) for old, now in zip(saved, snapshots))
    assert all(old is now for old, now in zip(records, model.core.ffn.aux_records))
    aux = model.pop_moe_aux()
    assert int(aux["counts"][0].sum()) == 3 * 2 * model.config.latent_slots * 2
    assert model.pop_moe_aux()["loss"] is None


def test_hypothesis_groups_isolate_gradients_until_final_merge_even_with_core_moe():
    model = core_moe_model(slot_groups=2).train()
    x = torch.randn(1, 5, model.config.hidden_size)
    valid = torch.ones(1, 5, dtype=torch.bool)
    _, states = model.reason(x, valid, steps=3, return_states=True)
    weights = torch.randn_like(states[0][:, :2])
    early = torch.autograd.grad((states[2][:, :2] * weights).sum(), model.core.slots, retain_graph=True)[0]
    final = torch.autograd.grad((states[3][:, :2] * weights).sum(), model.core.slots)[0]
    assert torch.count_nonzero(early[2:]) == 0
    assert final[2:].abs().sum() > 0
