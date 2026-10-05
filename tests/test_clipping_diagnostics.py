"""Combined-update clipping counts, bounded observations and exact recovery."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nuri import trainer
from nuri.config import tiny_config
from nuri.data import build_instruction_batch
from nuri.model import NuriModel
from nuri.trainer import Task, TrainSettings, TrainingInterrupted, train


def fresh_run(tmp_path, name, **overrides):
    torch.manual_seed(19)
    config = tiny_config()
    model = NuriModel(config)
    generator = torch.Generator().manual_seed(11)
    ids = torch.tensor([[5, 6, config.mask_id, 8], [9, config.mask_id, 11, 12]])
    batch = {"input_ids": ids, "input_valid": torch.ones_like(ids, dtype=torch.bool),
             "labels": torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]),
             "target_mask": ids == config.mask_id}
    tasks = {
        "mlm": Task("mlm", lambda: batch),
        "span": Task("span", lambda: build_instruction_batch(
            [([5, 6, 7], [8, 9, 10]), ([11, 12], [13, 14])], config, generator),
            generator=generator),
    }
    settings = TrainSettings(str(tmp_path / name), total_steps=6, warmup_steps=1,
                             log_every=1, grad_accum=2, mix={"mlm": 0.5, "span": 0.5},
                             seed=11, progress=False)
    return model, tasks, replace(settings, **overrides)


def assert_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def final_state(settings):
    return torch.load(Path(settings.out_dir) / "training_state/final.pt", weights_only=False)


def test_chunks_bound_noncontiguous_views_without_copies():
    values = torch.arange(66, dtype=torch.float32).reshape(3, 2, 11).transpose(0, 2)
    assert not values.is_contiguous()
    parts = list(trainer._tensor_chunks(values, 7))
    assert all(part.numel() <= 7 for part in parts)
    assert all(part.untyped_storage().data_ptr() == values.untyped_storage().data_ptr() for part in parts)
    assert sum(part.numel() for part in parts) == values.numel()
    assert sum(float(part.sum()) for part in parts) == float(values.sum())


def test_disjoint_fp32_norms_include_tied_weight_once_and_do_not_mutate():
    torch.manual_seed(7)
    model = NuriModel(tiny_config()).bfloat16()
    model.vocabulary_alias = model.embedding  # another reference to the tied parameter
    groups = trainer._gradient_parameter_groups(model)
    identities = [id(parameter) for parameters in groups.values() for parameter in parameters]
    assert len(identities) == len(set(identities)) == len(list(model.parameters()))
    assert groups["tied_embedding"] == [model.embedding.weight]
    expected = {}
    for name, parameters in groups.items():
        parameter = next(parameter for parameter in parameters if parameter.ndim == 2)
        gradient = (torch.arange(parameter.numel()).reshape(parameter.shape[1], parameter.shape[0]) % 11 - 5).t()
        parameter.grad = gradient.to(parameter.dtype)
        assert not parameter.grad.is_contiguous()
        expected[name] = float(parameter.grad.float().norm())
    before = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
              if parameter.grad is not None}
    rng = torch.get_rng_state().clone()
    observed = trainer._sample_optimizer_diagnostics(model, groups, chunk_elements=128)
    assert observed["pre_clip_grad_norms"] == pytest.approx(expected, rel=1e-6)
    rows = model.readout.position_embedding.weight.detach().float().norm(dim=1)
    assert observed["canvas_position_row_norms"] == pytest.approx({
        "min": float(rows.min()), "mean": float(rows.mean()), "median": float(rows.quantile(0.5)),
        "max": float(rows.max()), "rows": rows.numel(),
    })
    assert torch.equal(rng, torch.get_rng_state())
    for name, parameter in model.named_parameters():
        if name in before:
            assert torch.equal(parameter.grad, before[name])


def test_legacy_positions_are_omitted_and_unused_gradients_are_zero():
    model = NuriModel(tiny_config(canvas_position_encoding="none"))
    observed = trainer._sample_optimizer_diagnostics(model, trainer._gradient_parameter_groups(model))
    assert observed == {"pre_clip_grad_norms": {
        "encoder": 0.0, "core": 0.0, "readout": 0.0, "tied_embedding": 0.0, "other": 0.0,
    }}


@pytest.fixture
def controlled_updates(tmp_path, monkeypatch):
    """Microbatch cancellation and epsilon boundary distinguish real update counts."""
    model, tasks, settings = fresh_run(tmp_path, "controlled", mix={"mlm": 1.0},
                                       warmup_steps=0, log_every=4)
    coefficients = iter([3, -1, 0.2, 0.2, 2, 2, -0.5, 0.5, 1.5, 1.5, -2, -2])
    sampled, clipped = [], []
    original_sample = trainer._sample_optimizer_diagnostics
    original_clip = torch.nn.utils.clip_grad_norm_

    def controlled_loss(model, *args):
        return {"loss": model.embedding.weight[0, 0] * next(coefficients), "tokens": 8, "targets": 2}

    def sample(*args, **kwargs):
        sampled.append(len(clipped))
        return original_sample(*args, **kwargs)

    def clip(parameters, threshold):
        parameters = list(parameters)
        before = model.embedding.weight.grad.clone()
        norm = original_clip(parameters, threshold)
        coefficient = torch.clamp(threshold / (norm + 1e-6), max=1.0)
        assert torch.equal(model.embedding.weight.grad, before * coefficient)
        clipped.append(float(coefficient) < 1.0)
        return norm

    monkeypatch.setattr(trainer, "_task_loss", controlled_loss)
    monkeypatch.setattr(trainer, "_sample_optimizer_diagnostics", sample)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    records = train(model, tasks, settings, device=torch.device("cpu"))
    return records, sampled, clipped, settings


def test_clip_fraction_counts_unsampled_combined_updates(controlled_updates):
    records, sampled, clipped, settings = controlled_updates
    assert sampled == [0, 4, 5]
    assert clipped == [True, False, True, False, True, True]
    assert [record["optimizer_clipping"]["completed_updates"] for record in records] == [1, 5, 6]
    assert [record["optimizer_clipping"]["clipped_updates"] for record in records] == [1, 3, 4]
    assert [record["optimizer_clipping"]["fraction"] for record in records] == pytest.approx([1, 3 / 5, 4 / 6])
    assert [record["pre_clip_grad_norms"]["tied_embedding"] for record in records] == pytest.approx([1, 1.5, 2])
    assert all(record["pre_clip_grad_norms"]["readout"] == 0 for record in records)
    assert final_state(settings)["extra"]["totals"]["optimizer_clipping"] == {
        "start_completed_step": 0, "completed_updates": 6, "clipped_updates": 4,
    }


def test_real_events_match_counts_and_sampled_numeric_diagnostics(controlled_updates):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from nuri.monitoring import TrainingMonitor

    records, _, _, settings = controlled_updates
    event_dir = Path(settings.out_dir) / "events"
    with TrainingMonitor(6, 0, 0, progress=False, tensorboard_dir=event_dir) as monitor:
        for record in records:
            monitor.record(record, "mlm", record["batch_summary"]["targets"], "stable")
    events = EventAccumulator(str(event_dir), size_guidance={"scalars": 0}).Reload()
    for tag, key in {"clip_count": "clipped_updates", "clip_observed_updates": "completed_updates",
                     "clip_fraction": "fraction", "clip_window_start_step": "start_completed_step",
                     "clip_applied": "clip_applied", "clip_threshold": "threshold"}.items():
        points = events.Scalars(f"optimizer/{tag}")
        assert [point.step for point in points] == [1, 5, 6]
        assert [point.value for point in points] == pytest.approx([float(record["optimizer_clipping"][key]) for record in records])
    for group in ("encoder", "core", "readout", "tied_embedding", "other"):
        assert [point.value for point in events.Scalars(f"optimizer/pre_clip_grad_norm/{group}")] == pytest.approx(
            [record["pre_clip_grad_norms"][group] for record in records])
    for statistic in ("min", "mean", "median", "max", "rows"):
        assert [point.value for point in events.Scalars(f"parameters/canvas_position_row_norm/{statistic}")] == pytest.approx(
            [record["canvas_position_row_norms"][statistic] for record in records])
    assert not events.Tags()["tensors"]  # numeric only, no input tokens or paths


def test_sample_cadence_and_diagnostics_do_not_change_weights_optimizer_or_rng(tmp_path, monkeypatch):
    model, tasks, observed = fresh_run(tmp_path, "observed")
    train(model, tasks, observed, device=torch.device("cpu"))
    with monkeypatch.context() as context:
        context.setattr(trainer, "_sample_optimizer_diagnostics", lambda *args: {})
        model, tasks, plain = fresh_run(tmp_path, "plain", log_every=4)
        train(model, tasks, plain, device=torch.device("cpu"))
    left, right = final_state(observed), final_state(plain)
    for key in ("model", "optimizer", "torch_rng", "python_rng"):
        assert_equal(left[key], right[key])
    assert_equal(left["extra"]["random"], right["extra"]["random"])
    assert_equal(left["extra"]["totals"], right["extra"]["totals"])


@pytest.mark.parametrize("legacy", [False, True])
def test_clipping_history_resume_and_legacy_window(tmp_path, legacy):
    model, tasks, interrupted = fresh_run(tmp_path, "interrupted", stop_after_completed_steps=3)
    with pytest.raises(TrainingInterrupted):
        train(model, tasks, interrupted, device=torch.device("cpu"))
    checkpoint = Path(interrupted.out_dir) / "training_state/latest.pt"
    previous = torch.load(checkpoint, weights_only=False)
    if legacy:
        del previous["extra"]["totals"]["optimizer_clipping"]
        checkpoint = tmp_path / "legacy.pt"
        torch.save(previous, checkpoint)
    model, tasks, resumed = fresh_run(tmp_path, "resumed")
    records = train(model, tasks, resumed, device=torch.device("cpu"), resume_from=checkpoint)
    model, tasks, full = fresh_run(tmp_path, "full")
    full_records = train(model, tasks, full, device=torch.device("cpu"))
    left, right = final_state(full), final_state(resumed)
    for key in ("model", "optimizer", "torch_rng", "python_rng"):
        assert_equal(left[key], right[key])
    assert_equal(left["extra"]["random"], right["extra"]["random"])
    clipping = right["extra"]["totals"]["optimizer_clipping"]
    if legacy:
        assert clipping["start_completed_step"] == 3
        assert clipping["completed_updates"] == 3
        assert clipping["clipped_updates"] == sum(record["optimizer_clipping"]["clip_applied"] for record in records)
        assert [record["optimizer_clipping"]["completed_updates"] for record in records] == [1, 2, 3]
    else:
        assert_equal(left["extra"]["totals"], right["extra"]["totals"])
        for expected, actual in zip(full_records[3:], records):
            assert {key: value for key, value in expected.items() if key != "seconds"} == {
                key: value for key, value in actual.items() if key != "seconds"}


def test_nonfinite_attempt_is_not_counted_or_promoted(tmp_path, monkeypatch):
    model, tasks, settings = fresh_run(tmp_path, "nonfinite", mix={"mlm": 1.0}, save_every=1)
    calls = iter([2.0, 2.0, float("nan"), 2.0])

    def loss(model, *args):
        return {"loss": model.embedding.weight[0, 0] * next(calls), "tokens": 8, "targets": 2}

    monkeypatch.setattr(trainer, "_task_loss", loss)
    with pytest.raises(trainer.NonFiniteTraining):
        train(model, tasks, settings, device=torch.device("cpu"))
    saved = torch.load(Path(settings.out_dir) / "training_state/step-1.pt", weights_only=False)
    assert saved["extra"]["totals"]["optimizer_clipping"]["completed_updates"] == 1
    assert saved["extra"]["totals"]["optimizer_clipping"]["clipped_updates"] == 1
    assert_equal(saved["model"], model.state_dict())
    assert not (Path(settings.out_dir) / "training_state/final.pt").exists()

