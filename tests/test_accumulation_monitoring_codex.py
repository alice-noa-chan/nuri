"""Independent accumulation accounting and checkpoint-continuation regressions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nuri import trainer
from nuri.config import tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.model import NuriModel
from nuri.monitoring import TrainingMonitor
from nuri.pipeline import base_stage
from nuri.trainer import TrainSettings, train


def _fresh_run(tmp_path, name, accumulation=1):
    corpus_dir = tmp_path / "corpus"
    if not corpus_dir.exists():
        write_corpus(corpus_dir, [[3 + (row + pos) % 90 for pos in range(24 + row % 12)]
                                 for row in range(20)])
    config = tiny_config()
    torch.manual_seed(19)
    model = NuriModel(config)
    tasks, _ = base_stage(TokenCorpus(corpus_dir), config, batch_size=3, seed=11)
    assert tasks["mlm"].rng is tasks["span"].rng
    assert tasks["mlm"].generator is tasks["span"].generator
    settings = TrainSettings(
        out_dir=str(tmp_path / name), total_steps=6, warmup_steps=1, peak_lr=2e-3,
        save_every=3, log_every=1, seed=11, mix={"span": 1.0},
        grad_accum=accumulation, progress=False,
    )
    return model, tasks, settings


@pytest.fixture
def observed_accumulation(tmp_path, monkeypatch):
    """Capture real per-microbatch losses and the values given to the monitor."""
    model, tasks, settings = _fresh_run(tmp_path, "observed", accumulation=3)
    components, monitored_targets = [], []
    original_loss = trainer._task_loss
    original_record = TrainingMonitor.record

    def capture_loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        components.append(float(result["denoise"]))
        return result

    def capture_record(self, record, kind, batch_targets, schedule_phase):
        monitored_targets.append(batch_targets)
        original_record(self, record, kind, batch_targets, schedule_phase)

    monkeypatch.setattr(trainer, "_task_loss", capture_loss)
    monkeypatch.setattr(TrainingMonitor, "record", capture_record)
    records = train(model, tasks, settings, device=torch.device("cpu"))
    return records, components, monitored_targets


def test_dense_span_component_matches_the_optimized_microbatch_mean(observed_accumulation):
    """A pure denoising step must report the same objective in both loss fields."""
    records, components, _ = observed_accumulation
    for index, record in enumerate(records):
        expected = sum(components[3 * index:3 * index + 3]) / 3
        assert record["loss"] == pytest.approx(expected, abs=1e-6)
        assert record["denoise"] == pytest.approx(expected, abs=1e-6)


def test_monitor_targets_include_every_microbatch(observed_accumulation):
    """The monitor's batch count must equal all targets in that optimizer update."""
    records, _, monitored_targets = observed_accumulation
    previous = 0
    for record, reported in zip(records, monitored_targets):
        assert reported == record["targets"] - previous
        previous = record["targets"]


class PlannedInterruption(RuntimeError):
    """Interrupt immediately after the genuine midpoint state has been written."""


def _middle_checkpoint(tmp_path, monkeypatch, accumulation):
    model, tasks, settings = _fresh_run(tmp_path, "interrupted", accumulation)
    original_save = trainer._maybe_save

    def save_then_interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[4] == 2:
            raise PlannedInterruption

    with monkeypatch.context() as context:
        context.setattr(trainer, "_maybe_save", save_then_interrupt)
        with pytest.raises(PlannedInterruption):
            train(model, tasks, settings, device=torch.device("cpu"))
    checkpoint = Path(settings.out_dir) / "training_state" / "step-3.pt"
    assert checkpoint.is_file()
    return settings, checkpoint


@pytest.mark.parametrize("requested_accumulation", [1, 2])
def test_legacy_settings_default_to_one_microbatch(tmp_path, monkeypatch, requested_accumulation):
    """Pre-accumulation states resume at one microbatch and reject a changed count."""
    settings, checkpoint = _middle_checkpoint(tmp_path, monkeypatch, accumulation=1)
    state = torch.load(checkpoint, weights_only=False)
    del state["extra"]["settings"]["grad_accum"]
    torch.save(state, checkpoint)
    model, tasks, _ = _fresh_run(tmp_path, "resumed", requested_accumulation)
    resumed_settings = replace(settings, out_dir=str(tmp_path / "resumed"),
                               grad_accum=requested_accumulation)
    if requested_accumulation == 1:
        records = train(model, tasks, resumed_settings, device=torch.device("cpu"),
                        resume_from=checkpoint)
        assert [record["step"] for record in records] == [3, 4, 5]
    else:
        with pytest.raises(ValueError, match="grad_accum"):
            train(model, tasks, resumed_settings, device=torch.device("cpu"),
                  resume_from=checkpoint)


def _assert_equal(expected, actual):
    if torch.is_tensor(expected):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_equal(left, right)
    else:
        assert expected == actual


def test_accumulated_resume_preserves_optimizer_and_complete_json_trace(tmp_path, monkeypatch):
    """Three-microbatch updates must continue exactly across a saved boundary."""
    settings, checkpoint = _middle_checkpoint(tmp_path, monkeypatch, accumulation=3)
    full_model, full_tasks, full_settings = _fresh_run(tmp_path, "full", accumulation=3)
    train(full_model, full_tasks, full_settings, device=torch.device("cpu"))
    resumed_model, resumed_tasks, resumed_settings = _fresh_run(tmp_path, "resumed", accumulation=3)
    train(resumed_model, resumed_tasks, resumed_settings, device=torch.device("cpu"),
          resume_from=checkpoint)

    def read_records(folder):
        return [{k: v for k, v in json.loads(line).items() if k != "seconds"}
                for line in (Path(folder) / "train_log.jsonl").read_text().splitlines()]

    assert read_records(full_settings.out_dir) == (
        read_records(settings.out_dir) + read_records(resumed_settings.out_dir))
    _assert_equal(full_model.state_dict(), resumed_model.state_dict())
    full_state = torch.load(Path(full_settings.out_dir) / "training_state" / "final.pt", weights_only=False)
    resumed_state = torch.load(Path(resumed_settings.out_dir) / "training_state" / "final.pt", weights_only=False)
    _assert_equal(full_state["optimizer"], resumed_state["optimizer"])
    _assert_equal(full_state["extra"]["totals"], resumed_state["extra"]["totals"])
    _assert_equal(full_state["extra"]["random"], resumed_state["extra"]["random"])
