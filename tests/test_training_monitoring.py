"""Check real training output, TensorBoard files, cleanup and exact continuation."""

import builtins
from dataclasses import replace

import pytest
import torch

from nuri import trainer
from nuri.config import MoEConfig, tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.model import NuriModel
from nuri.monitoring import TrainingMonitor
from nuri.pipeline import base_stage
from nuri.trainer import TrainSettings, train


def fresh_run(tmp_path, name="run", moe=False):
    corpus_dir = tmp_path / "corpus"
    if not corpus_dir.exists():
        write_corpus(corpus_dir, [[3 + (row + pos) % 70 for pos in range(20)] for row in range(12)])
    config = tiny_config(moe=MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2,
                                       input_layers=(2,), readout_layers=(2,)) if moe else None)
    torch.manual_seed(19)
    model = NuriModel(config)
    tasks, _ = base_stage(TokenCorpus(corpus_dir), config, batch_size=3, seed=7)
    settings = TrainSettings(str(tmp_path / name), total_steps=6, warmup_steps=1,
                             log_every=2, save_every=3, seed=7, mix={"span": 1.0},
                             stage="base", progress=False)
    return model, tasks, settings


def read_events(folder):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(str(folder), size_guidance={"scalars": 0}).Reload()


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


@pytest.mark.parametrize("moe", [False, True])
def test_events_match_sampled_training_metrics_without_changing_training(tmp_path, moe):
    pytest.importorskip("tensorboard")
    model, tasks, settings = fresh_run(tmp_path, "plain", moe)
    plain_records = train(model, tasks, settings, device=torch.device("cpu"))

    observed, tasks, settings = fresh_run(tmp_path, "observed", moe)
    event_dir = tmp_path / "events"
    settings.tensorboard_dir = str(event_dir)
    records = train(observed, tasks, settings, device=torch.device("cpu"))
    for left, right in zip(plain_records, records):
        assert {k: v for k, v in left.items() if k != "seconds"} == {
            k: v for k, v in right.items() if k != "seconds"}
    assert_equal(model.state_dict(), observed.state_dict())
    plain_state = torch.load(tmp_path / "plain/training_state/final.pt", weights_only=False)
    observed_state = torch.load(tmp_path / "observed/training_state/final.pt", weights_only=False)
    assert_equal(plain_state["optimizer"], observed_state["optimizer"])
    assert_equal(plain_state["extra"]["random"], observed_state["extra"]["random"])

    events = read_events(event_dir)
    for tag, key in {"loss/total": "loss", "loss/span/denoise": "denoise",
                     "optimizer/lr": "lr", "optimizer/grad_norm": "grad_norm",
                     "data/processed_tokens": "tokens"}.items():
        points = events.Scalars(tag)
        assert [p.step for p in points] == [1, 3, 5, 6]
        assert [p.value for p in points] == pytest.approx([r[key] for r in records])
    assert all(p.value > 0 for p in events.Scalars("performance/tokens_per_second"))
    assert [p.value for p in events.Scalars("train/schedule_phase")] == [0, 1, 1, 2]
    assert "batch/span/targets" in events.Tags()["scalars"]
    assert not events.Tags()["tensors"]  # no text, arguments, paths or input tokens
    if moe:
        assert [p.value for p in events.Scalars("moe/balance")] == pytest.approx([r["moe_balance"] for r in records])
        for layer, counts in enumerate(records[-1]["expert_counts"]):
            for expert, count in enumerate(counts):
                assert events.Scalars(f"moe/layer_{layer}/expert_{expert}_assignments")[-1].value == count


def test_progress_identifies_operations_metrics_and_saved_checkpoint(tmp_path, capsys, monkeypatch):
    model, tasks, settings = fresh_run(tmp_path)
    settings.progress = True
    original_phase = TrainingMonitor.phase

    # Force rendering in this short CPU run; normal runs throttle it to 2 Hz.
    def visible_phase(self, operation, force=False):
        original_phase(self, operation, force=True)

    monkeypatch.setattr(TrainingMonitor, "phase", visible_phase)
    train(model, tasks, settings, device=torch.device("cpu"))
    output = capsys.readouterr().err
    for text in ("base/span", "warmup", "stable", "decay", "batch", "forward", "backward",
                 "optimizer", "saving step-3", "saving final", "loss=", "lr=", "grad=", "tok/s=", "6/6", "done"):
        assert text in output


def test_resume_purges_stale_events_and_preserves_saved_progress(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    model, tasks, settings = fresh_run(tmp_path)
    settings.log_every = 1
    settings.tensorboard_dir = str(tmp_path / "events")
    full_records = train(model, tasks, settings, device=torch.device("cpu"))
    original_monitor = trainer.TrainingMonitor
    monitors = []

    class CaptureMonitor(original_monitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            monitors.append(self)

    monkeypatch.setattr(trainer, "TrainingMonitor", CaptureMonitor)
    resumed, tasks, _ = fresh_run(tmp_path, "resumed")
    resumed_settings = replace(settings, out_dir=str(tmp_path / "resumed"), progress=True)
    records = train(resumed, tasks, resumed_settings, device=torch.device("cpu"),
                    resume_from=tmp_path / "run/training_state/step-3.pt")
    assert monitors[0].bar.initial == 3
    assert monitors[0].bar.n == 6
    assert [r["step"] for r in records] == [3, 4, 5]
    points = read_events(tmp_path / "events").Scalars("loss/total")
    assert [p.step for p in points] == [1, 2, 3, 4, 5, 6]  # stale 4..6 were replaced, not duplicated
    assert [p.value for p in points] == pytest.approx([r["loss"] for r in full_records])
    assert_equal(model.state_dict(), resumed.state_dict())
    finished, tasks, _ = fresh_run(tmp_path, "finished")
    assert train(finished, tasks, replace(settings, out_dir=str(tmp_path / "finished")),
                 device=torch.device("cpu"), resume_from=tmp_path / "run/training_state/final.pt") == []
    assert [p.step for p in read_events(tmp_path / "events").Scalars("loss/total")] == [1, 2, 3, 4, 5, 6]


def test_resumed_throughput_counts_only_new_tokens(monkeypatch):
    class Clock:
        now = 10.0

    clock = Clock()
    monkeypatch.setattr("nuri.monitoring.time.perf_counter", lambda: clock.now)
    with TrainingMonitor(100, 50, 1_000_000, progress=True) as monitor:
        clock.now += 2
        monitor.record({"step": 51, "task": "mlm", "depth": 4, "loss": 2.0,
                        "lr": 1e-3, "grad_norm": 1.0, "tokens": 1_000_200}, "mlm", 20, "stable")
        assert "tok/s=100" in monitor.bar.postfix


def test_training_failure_flushes_events_and_closes_progress(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    model, tasks, settings = fresh_run(tmp_path)
    settings.tensorboard_dir = str(tmp_path / "events")
    settings.progress = True
    original_batch = tasks["span"].next_batch
    monitors = []

    class CaptureMonitor(TrainingMonitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            monitors.append(self)

    calls = 0

    def interrupted_batch():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate batch interruption")
        return original_batch()

    tasks["span"].next_batch = interrupted_batch
    monkeypatch.setattr(trainer, "TrainingMonitor", CaptureMonitor)
    with pytest.raises(RuntimeError, match="deliberate batch interruption"):
        train(model, tasks, settings, device=torch.device("cpu"))
    monitor = monitors[0]
    assert monitor.bar.n == 1
    assert monitor.bar.disable  # close() disables the display
    assert "interrupted" in monitor.bar.desc
    assert monitor.writer.file_writer is None
    assert [p.step for p in read_events(tmp_path / "events").Scalars("loss/total")] == [1]


def test_missing_tensorboard_fails_clearly_only_when_enabled(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_tensorboard(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tensorboard)
    with TrainingMonitor(1, 0, 0, progress=False):
        pass
    with pytest.raises(RuntimeError, match=r'pip install.*tensorboard'):
        TrainingMonitor(1, 0, 0, progress=False, tensorboard_dir=tmp_path / "events")


def test_invalid_resume_does_not_purge_event_history(tmp_path):
    pytest.importorskip("tensorboard")
    model, tasks, settings = fresh_run(tmp_path)
    settings.tensorboard_dir = str(tmp_path / "events")
    train(model, tasks, settings, device=torch.device("cpu"))
    previous_files = sorted((tmp_path / "events").iterdir())
    model, tasks, _ = fresh_run(tmp_path, "invalid")
    with pytest.raises(ValueError, match="peak_lr"):
        train(model, tasks, replace(settings, peak_lr=1e-2), device=torch.device("cpu"),
              resume_from=tmp_path / "run/training_state/step-3.pt")
    assert sorted((tmp_path / "events").iterdir()) == previous_files
    assert [p.step for p in read_events(tmp_path / "events").Scalars("loss/total")] == [1, 3, 5, 6]


@pytest.mark.parametrize("moe", [False, True])
def test_accumulated_depths_and_mixed_task_losses_write_numeric_events(tmp_path, monkeypatch, capsys, moe):
    pytest.importorskip("tensorboard")
    model, tasks, settings = fresh_run(tmp_path, moe=moe)
    settings.grad_accum = 3
    settings.mix = {"mlm": 0.5, "span": 0.5}
    settings.progress = True
    settings.tensorboard_dir = str(tmp_path / "events")
    counts_per_microbatch = []
    active_depths = []
    original_aux = model.pop_moe_aux
    original_begin = TrainingMonitor.begin_step

    def capture_begin(self, task, depth, schedule_phase, **kwargs):
        active_depths.append(depth)
        original_begin(self, task, depth, schedule_phase, **kwargs)

    def capture_aux():
        aux = original_aux()
        counts_per_microbatch.append([count.tolist() for count in aux["counts"]])
        return aux

    monkeypatch.setattr(model, "pop_moe_aux", capture_aux)
    monkeypatch.setattr(TrainingMonitor, "begin_step", capture_begin)
    records = train(model, tasks, settings, device=torch.device("cpu"))
    output = capsys.readouterr().err
    for text in ("micro=1/3", "micro=2/3", "micro=3/3"):
        assert text in output
    assert any(isinstance(record["depth"], list) for record in records)
    assert any("+" in record["task"] for record in records)
    events = read_events(tmp_path / "events")
    assert [p.step for p in events.Scalars("loss/total")] == [1, 3, 5, 6]
    for record in records:
        depths = active_depths[3 * record["step"]:3 * (record["step"] + 1)]
        expected = depths[0] if len(set(depths)) == 1 else depths
        assert record["depth"] == expected
    for key, reducer in {"mean": lambda depths: sum(depths) / len(depths), "min": min, "max": max}.items():
        expected = {record["step"] + 1: reducer(record["depth"])
                    for record in records if isinstance(record["depth"], list) and record["task"] != "mlm"}
        points = events.Scalars(f"train/reasoning_depth_{key}")
        assert {point.step: point.value for point in points} == pytest.approx(expected)
    mixed_steps = [record["step"] + 1 for record in records if "+" in record["task"]]
    assert [p.step for p in events.Scalars("loss/mixed/total")] == mixed_steps
    for task in ("mlm", "span"):
        if f"loss/{task}/total" in events.Tags()["scalars"]:
            assert not (set(p.step for p in events.Scalars(f"loss/{task}/total")) & set(mixed_steps))
    if moe:
        for record in records:
            micro_counts = counts_per_microbatch[3 * record["step"]:3 * (record["step"] + 1)]
            expected = [[sum(counts[layer][expert] for counts in micro_counts if layer < len(counts))
                         for expert in range(4)] for layer in range(max(map(len, micro_counts)))]
            assert record["expert_counts"] == expected
            for layer, counts in enumerate(expected):
                for expert, count in enumerate(counts):
                    points = events.Scalars(f"moe/layer_{layer}/expert_{expert}_assignments")
                    assert next(point.value for point in points if point.step == record["step"] + 1) == count
