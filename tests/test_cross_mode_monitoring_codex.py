"""Cross-mode loss reporting during real accumulated embedding training."""

import json
import math
from pathlib import Path

import pytest
import torch

from nuri import trainer
from nuri.config import tiny_config
from nuri.model import NuriModel
from nuri.monitoring import TrainingMonitor
from nuri.pipeline import embedding_stage
from nuri.trainer import TrainSettings, train


def fresh_embedding_run(tmp_path, name, events=False):
    config = tiny_config(hidden_size=16, num_heads=2, ffn_size=32,
                         input_layers=1, readout_layers=1, embed_dim=8)
    torch.manual_seed(19)
    model = NuriModel(config)
    pairs = [([3 + row, 30 + row, 60 + row], [30 + row, 3 + row, 60 + row]) for row in range(12)]
    tasks, mix = embedding_stage(pairs, config, batch_size=3, seed=11)
    settings = TrainSettings(
        str(tmp_path / name), total_steps=4, warmup_steps=1, peak_lr=2e-3,
        seed=11, mix=mix, grad_accum=3, log_every=1, progress=False, stage="embed",
        embed_dims=(4, 8), cross_mode_weight=0.4,
        tensorboard_dir=str(tmp_path / name / "events") if events else None,
    )
    return model, tasks, settings


def read_events(folder):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(str(folder), size_guidance={"scalars": 0}).Reload()


def assert_state_equal(expected, actual):
    if torch.is_tensor(expected):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_state_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_state_equal(left, right)
    else:
        assert expected == actual


def test_accumulated_cross_mode_component_is_the_raw_microbatch_mean(tmp_path, monkeypatch):
    model, tasks, settings = fresh_embedding_run(tmp_path, "accumulated")
    components = []
    original = trainer._task_loss

    def capture_loss(*args, **kwargs):
        result = original(*args, **kwargs)
        components.append({key: float(result[key].detach()) for key in ("fast", "reasoning", "cross_mode", "loss")})
        return result

    monkeypatch.setattr(trainer, "_task_loss", capture_loss)
    records = train(model, tasks, settings, device=torch.device("cpu"))
    assert len(components) == settings.total_steps * settings.grad_accum
    assert len({part["cross_mode"] for part in components}) > 1
    for index, record in enumerate(records):
        micro = components[index * 3:(index + 1) * 3]
        means = {key: sum(part[key] for part in micro) / 3 for key in micro[0]}
        for key, expected in means.items():
            assert record[key] == pytest.approx(expected, abs=2e-6)
        expected = (means["fast"] + means["reasoning"]) / 2 + settings.cross_mode_weight * means["cross_mode"]
        assert record["loss"] == pytest.approx(expected, abs=2e-6)
        assert record["cross_mode"] > 0
    saved = [json.loads(line) for line in (Path(settings.out_dir) / "train_log.jsonl").read_text().splitlines()]
    assert saved == records


def test_numeric_cross_mode_events_and_terminal_postfix_match_training_records(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    model, tasks, settings = fresh_embedding_run(tmp_path, "observed", events=True)
    postfixes = []
    original = TrainingMonitor.record

    def capture_record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        postfixes.append(self.bar.postfix)

    monkeypatch.setattr(TrainingMonitor, "record", capture_record)
    records = train(model, tasks, settings, device=torch.device("cpu"))
    events = read_events(settings.tensorboard_dir)
    tags = events.Tags()
    for key in ("fast", "reasoning", "cross_mode"):
        points = events.Scalars(f"loss/pairs/{key}")
        assert [point.step for point in points] == [1, 2, 3, 4]
        assert [point.value for point in points] == pytest.approx([record[key] for record in records])
    assert len(postfixes) == len(records)
    for postfix, record in zip(postfixes, records):
        assert f"cross_mode={record['cross_mode']:.4f}" in postfix
    for tag in tags["scalars"]:
        assert all(math.isfinite(point.value) for point in events.Scalars(tag))
    for kind in ("tensors", "images", "audio", "histograms"):
        assert not tags[kind]


def test_cross_mode_event_logging_preserves_weights_optimizer_and_random_state(tmp_path):
    pytest.importorskip("tensorboard")
    plain_model, tasks, settings = fresh_embedding_run(tmp_path, "plain")
    plain_records = train(plain_model, tasks, settings, device=torch.device("cpu"))
    observed_model, tasks, settings = fresh_embedding_run(tmp_path, "with-events", events=True)
    observed_records = train(observed_model, tasks, settings, device=torch.device("cpu"))
    for plain, observed in zip(plain_records, observed_records):
        assert {key: value for key, value in plain.items() if key != "seconds"} == {
            key: value for key, value in observed.items() if key != "seconds"
        }
    assert_state_equal(plain_model.state_dict(), observed_model.state_dict())
    plain = torch.load(tmp_path / "plain" / "training_state" / "final.pt", weights_only=False)
    observed = torch.load(tmp_path / "with-events" / "training_state" / "final.pt", weights_only=False)
    assert_state_equal(plain["optimizer"], observed["optimizer"])
    assert_state_equal(plain["extra"]["random"], observed["extra"]["random"])
    assert_state_equal(plain["extra"]["totals"], observed["extra"]["totals"])


def test_legacy_missing_cross_mode_weight_resumes_zero_exactly_and_rejects_nonzero(tmp_path):
    full_model, tasks, settings = fresh_embedding_run(tmp_path, "legacy-full")
    settings.cross_mode_weight = 0.0
    settings.save_every = 2
    full_records = train(full_model, tasks, settings, device=torch.device("cpu"))
    state = torch.load(tmp_path / "legacy-full" / "training_state" / "step-2.pt", weights_only=False)
    del state["extra"]["settings"]["cross_mode_weight"]
    legacy = tmp_path / "legacy.pt"
    torch.save(state, legacy)

    enabled, tasks, settings = fresh_embedding_run(tmp_path, "legacy-enabled")
    with pytest.raises(ValueError, match="cross_mode_weight"):
        train(enabled, tasks, settings, device=torch.device("cpu"), resume_from=legacy)

    resumed_model, tasks, settings = fresh_embedding_run(tmp_path, "legacy-resumed")
    settings.cross_mode_weight = 0.0
    resumed_records = train(resumed_model, tasks, settings, device=torch.device("cpu"), resume_from=legacy)
    assert [record["step"] for record in resumed_records] == [2, 3]
    for full, resumed in zip(full_records[2:], resumed_records):
        assert {key: value for key, value in full.items() if key != "seconds"} == {
            key: value for key, value in resumed.items() if key != "seconds"
        }
    assert all("cross_mode" not in record for record in resumed_records)
    assert_state_equal(full_model.state_dict(), resumed_model.state_dict())
    full = torch.load(tmp_path / "legacy-full" / "training_state" / "final.pt", weights_only=False)
    resumed = torch.load(tmp_path / "legacy-resumed" / "training_state" / "final.pt", weights_only=False)
    assert_state_equal(full["optimizer"], resumed["optimizer"])
    assert_state_equal(full["extra"]["random"], resumed["extra"]["random"])
    assert_state_equal(full["extra"]["totals"], resumed["extra"]["totals"])
