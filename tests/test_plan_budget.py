import importlib.util
import json
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "plan_budget.py"
    spec = importlib.util.spec_from_file_location("plan_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench(path: Path, seconds: float, batch=64, input_length=1024, output_length=None, training=True) -> str:
    path.write_text(json.dumps({"training": training, "batch_size": batch, "input_length": input_length,
                                "output_length": output_length, "latency_seconds": {"mean": seconds}}))
    return str(path)


def make_plan(tmp_path, mlm_seconds, span_seconds):
    plan = {
        "out": "runs/release",
        "stages": {
            "base-dense": {"stage": "base", "preset": "release-dense", "batch_size": 64},
            "it-dense": {"stage": "it", "init": "base-dense", "batch_size": 64, "grad_accum": 2},
        },
        "budget": {"overhead": 1.0, "stages": {
            "base-dense": {"hours": 8, "benchmarks": {
                "mlm": bench(tmp_path / "mlm.json", mlm_seconds),
                "span": bench(tmp_path / "span.json", span_seconds, output_length=256)}},
            "it-dense": {"hours": 1.5, "benchmarks": {
                "instruction": bench(tmp_path / "it.json", 0.5, input_length=512, output_length=256)}},
        }},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_steps_tokens_and_checkpoint_cadence(tmp_path):
    plan = make_plan(tmp_path, mlm_seconds=0.4, span_seconds=0.6)
    reports = load_script().main([str(plan), "--out", str(tmp_path / "out.json")])
    base, it = reports
    assert base["seconds_per_step"] == pytest.approx(0.5)  # 0.5 * 0.4 + 0.5 * 0.6
    assert base["steps"] == 8 * 3600 * 2  # 8 h / 0.5 s
    tokens_per_step = 0.5 * 64 * 1024 + 0.5 * 64 * (1024 + 256)
    assert base["processed_tokens"] == pytest.approx(base["steps"] * tokens_per_step)
    assert base["save_every"] == -(-1e9 // tokens_per_step)  # about every 1B tokens
    assert base["size_rule"] == "ok"
    assert it["seconds_per_step"] == pytest.approx(1.0)  # grad_accum 2 x 0.5 s
    written = json.loads((tmp_path / "out.json").read_text())
    assert written["stages"]["base-dense"]["steps"] == base["steps"]
    assert written["stages"]["it-dense"]["steps"] == it["steps"] and "save_every" not in written["stages"]["it-dense"]


def test_slow_gpu_triggers_the_size_rule(tmp_path):
    plan = make_plan(tmp_path, mlm_seconds=4.0, span_seconds=6.0)
    base = load_script().main([str(plan)])[0]
    assert base["processed_tokens"] < 3e9 and base["size_rule"].startswith("shrink")


def test_mismatched_batch_and_non_training_benchmarks(tmp_path):
    plan_path = make_plan(tmp_path, 0.4, 0.6)
    plan = json.loads(plan_path.read_text())
    plan["stages"]["base-dense"]["batch_size"] = 32
    plan_path.write_text(json.dumps(plan))
    assert load_script().main([str(plan_path)])[0]["warnings"]
    bench(tmp_path / "mlm.json", 0.4, training=False)
    with pytest.raises(ValueError, match="--training"):
        load_script().main([str(plan_path)])
