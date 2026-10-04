import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import first_hour_gate as gate  # noqa: E402

GIB = 1024 ** 3


def record(step, task, loss, seconds, valid=600, padded=1000, peak=None, grad=1.0):
    r = {"step": step, "task": task, "loss": loss, "grad_norm": grad, "seconds": seconds,
         "batch_summary": {"valid_input_tokens": valid, "padded_input_tokens": padded,
                           "valid_canvas_tokens": 0, "padded_canvas_tokens": 0, "rows": 4, "targets": 50}}
    if peak is not None:
        r["cuda_peak_allocated_bytes"] = peak
        r["cuda_peak_reserved_bytes"] = peak
    return r


def log(steps=60, seconds_per_step=0.5, start_loss=11.0, slope=-0.02, peak=40 * GIB):
    return [record(s, "mlm" if s % 2 else "span", start_loss + slope * s, 3.0 + s * seconds_per_step,
                   peak=peak) for s in range(steps)]


def test_steps_and_rates_from_a_steady_log():
    report = gate.analyze(log(), hours=10, skip=10, overhead=0.9, memory_limit_gib=76)
    assert report["passed"]
    assert report["mean_seconds_per_step"] == pytest.approx(0.5)
    assert report["plan"]["total_steps"] == int(10 * 3600 * 0.9 / 0.5)
    assert report["valid_positions_per_second"] == pytest.approx(600 / 0.5)
    assert report["valid_fraction"] == pytest.approx(0.6)
    assert set(report["per_task"]) == {"mlm", "span"}


def test_warmup_steps_are_not_timed():
    records = log()
    records[1]["seconds"] = 90.0  # a slow first compile step; later steps keep their spacing
    for r in records[2:]:
        r["seconds"] += 87.0 - 0.5
    report = gate.analyze(records, hours=1, skip=5)
    assert report["mean_seconds_per_step"] == pytest.approx(0.5)


def test_failures_are_reported():
    nan = log()
    nan[30]["loss"] = float("nan")
    assert gate.analyze(nan, hours=1)["checks"]["finite"] is False

    flat = log(slope=0.01)  # loss rising
    assert gate.analyze(flat, hours=1)["checks"]["loss_falling"] is False

    big = log(peak=78 * GIB)
    report = gate.analyze(big, hours=1, memory_limit_gib=76)
    assert report["checks"]["memory"] is False and not report["passed"]

    with pytest.raises(ValueError, match="no timed steps"):
        gate.analyze(log(steps=10), hours=1, skip=50)


def test_a_resumed_log_keeps_the_last_copy_and_ignores_the_clock_reset(tmp_path):
    first = log(steps=40)
    resumed = [record(s, "mlm" if s % 2 else "span", 11.0 - 0.02 * s, 2.0 + (s - 30) * 0.5, peak=40 * GIB)
               for s in range(30, 60)]
    path = tmp_path / "train_log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in first + resumed), encoding="utf-8")
    records = gate.read_records(path)
    assert [r["step"] for r in records] == list(range(60))
    report = gate.analyze(records, hours=1, skip=10)
    assert report["mean_seconds_per_step"] == pytest.approx(0.5)  # the reset at step 30 is not a negative time


def test_cli_writes_json_and_fails_with_exit_code(tmp_path):
    path = tmp_path / "base" / "train_log.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(r) for r in log(peak=79 * GIB)), encoding="utf-8")
    with pytest.raises(SystemExit) as info:
        gate.main([str(path), "--hours", "2", "--memory-limit-gib", "76", "--output", str(tmp_path / "g.json")])
    assert info.value.code == 1
    assert json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))["checks"]["memory"] is False


def test_reads_a_real_training_log(tmp_path):
    """End to end on a tiny CPU run of the real trainer (log_every=1)."""
    from nuri.config import tiny_config
    from nuri.data import TokenCorpus, write_corpus
    from nuri.model import NuriModel
    from nuri.pipeline import base_stage
    from nuri.trainer import TrainSettings, train

    rng = random.Random(0)
    write_corpus(tmp_path / "corpus", [[3 + (rng.randint(3, 60) + i) % 120 for i in range(rng.randint(20, 40))]
                                       for _ in range(100)])
    config = tiny_config()
    torch.manual_seed(0)
    tasks, mix = base_stage(TokenCorpus(tmp_path / "corpus"), config, batch_size=4, pack_mlm=True, bucket=True)
    settings = TrainSettings(out_dir=str(tmp_path / "run"), total_steps=30, peak_lr=3e-3, warmup_steps=2,
                             log_every=1, mix=mix, progress=False)
    train(NuriModel(config), tasks, settings, device=torch.device("cpu"))
    report = gate.analyze(gate.read_records(tmp_path / "run" / "train_log.jsonl"), hours=1, skip=2, loss_window=3)
    assert report["measured_steps"] == 28 and report["checks"]["finite"]
    assert 0 < report["valid_fraction"] <= 1 and report["valid_positions_per_second"] > 0
    assert report["cuda_peak_reserved_gib"] is None and report["checks"]["memory"] is None  # CPU
