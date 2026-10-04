"""Variable batch geometry stays observable without changing training or resume."""

import json
import random
from dataclasses import replace

import pytest
import torch

from nuri.config import tiny_config
from nuri.data import build_instruction_batch, build_pair_batch
from nuri.model import NuriModel
from nuri.monitoring import TrainingMonitor
from nuri.trainer import Task, TrainSettings, _cuda_peaks, _reset_cuda_peaks, train


def variable_run(tmp_path, *, tensorboard=False, device="cpu"):
    config = tiny_config()
    rng = random.Random(11)
    generator = torch.Generator().manual_seed(11)
    batches = []

    def next_batch():
        examples = [([3] * rng.randrange(2, 20), [4] * rng.randrange(1, 10))
                    for _ in range(rng.randrange(1, 6))]
        batch = build_instruction_batch(examples, config, generator)
        batches.append(batch)
        return batch

    torch.manual_seed(7)
    model = NuriModel(config)
    task = Task("span", next_batch, rng, generator)
    settings = TrainSettings(str(tmp_path), total_steps=3, warmup_steps=1, grad_accum=2,
                             log_every=1, save_every=1, mix={"span": 1.0}, progress=False,
                             tensorboard_dir=str(tmp_path / "events") if tensorboard else None)
    records = train(model, {"span": task}, settings, device=torch.device(device))
    return model, task, settings, batches, records


def test_variable_microbatches_log_actual_masks_and_shapes(tmp_path):
    _, _, _, batches, records = variable_run(tmp_path)
    saved = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert saved == records
    assert len({batch["input_ids"].shape for batch in batches}) > 1
    for step, record in enumerate(records):
        selected = batches[2 * step:2 * step + 2]
        summary = record["batch_summary"]
        assert summary["microbatches"] == 2
        assert summary["rows"] == sum(batch["input_ids"].shape[0] for batch in selected)
        assert summary["max_rows"] == max(batch["input_ids"].shape[0] for batch in selected)
        assert summary["valid_input_tokens"] == sum(int(batch["input_valid"].sum()) for batch in selected)
        assert summary["valid_canvas_tokens"] == sum(int(batch["canvas_valid"].sum()) for batch in selected)
        assert summary["targets"] == sum(int(batch["loss_mask"].sum()) for batch in selected)
        assert summary["padded_input_tokens"] == sum(batch["input_ids"].numel() for batch in selected)
        assert summary["padded_canvas_tokens"] == sum(batch["canvas"].numel() for batch in selected)
        valid = summary["valid_input_tokens"] + summary["valid_canvas_tokens"]
        padded = summary["padded_input_tokens"] + summary["padded_canvas_tokens"]
        assert summary["valid_fraction"] == pytest.approx(valid / padded)
        assert len(record["batch_shapes"]) == 2
        for batch, shape in zip(selected, record["batch_shapes"]):
            assert shape["kind"] == "span"
            assert shape["input_width"] == batch["input_ids"].shape[1]
            assert shape["canvas_width"] == batch["canvas"].shape[1]
        assert not any(key.startswith("cuda_") for key in record)


def test_tensorboard_geometry_matches_json_and_progress(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    _, _, _, _, records = variable_run(tmp_path, tensorboard=True)
    events = EventAccumulator(str(tmp_path / "events")).Reload()
    for key in records[0]["batch_summary"]:
        points = events.Scalars(f"batch/{key}")
        assert [point.step for point in points] == [1, 2, 3]
        assert [point.value for point in points] == pytest.approx([record["batch_summary"][key] for record in records])
    assert not events.Tags()["tensors"]
    assert not any(tag.startswith("memory/") for tag in events.Tags()["scalars"])
    with TrainingMonitor(3, 0, 0, progress=True) as monitor:
        monitor.record(records[0], "span", records[0]["batch_summary"]["targets"], "warmup")
        assert "rows=" in monitor.bar.postfix and "shape=" in monitor.bar.postfix
        assert "valid=" in monitor.bar.postfix


def test_geometry_logging_preserves_exact_resume(tmp_path):
    model, task, settings, _, records = variable_run(tmp_path / "original")
    resumed = NuriModel(model.config)
    continuation = train(resumed, {"span": task}, replace(settings, out_dir=str(tmp_path / "resumed")),
                         device=torch.device("cpu"),
                         resume_from=tmp_path / "original/training_state/step-1.pt")
    for original, restored in zip(records[1:], continuation):
        assert {key: value for key, value in original.items() if key != "seconds"} == {
            key: value for key, value in restored.items() if key != "seconds"}
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in model.state_dict().items())


def test_pair_rows_count_once_and_input_views_count_twice(tmp_path):
    config = tiny_config()
    pairs = [([3, 4], [5, 6, 7]), ([8, 9, 10], [11]), ([12], [13, 14])]
    task = Task("pairs", lambda: build_pair_batch(pairs, config))
    settings = TrainSettings(str(tmp_path), total_steps=1, warmup_steps=0,
                             mix={"pairs": 1.0}, embed_dims=(config.embed_dim,), progress=False)
    record = train(NuriModel(config), {"pairs": task}, settings, device=torch.device("cpu"))[0]
    shape = record["batch_shapes"][0]
    assert shape["rows"] == 3 and shape["targets"] == 3
    assert shape["query_width"] == 3 and shape["document_width"] == 3
    assert shape["padded_input_tokens"] == 18
    assert shape["valid_input_tokens"] == 12
    assert record["batch_summary"]["valid_canvas_tokens"] == 0


def test_cpu_never_reads_or_resets_cuda_allocator(monkeypatch):
    def unexpected(*args):
        pytest.fail("CPU monitoring must not call the CUDA allocator")

    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    _reset_cuda_peaks(torch.device("cpu"))
    assert _cuda_peaks(torch.device("cpu")) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an actual CUDA allocator")
def test_cuda_training_reports_positive_allocator_peaks(tmp_path):
    _, _, _, _, records = variable_run(tmp_path, device="cuda")
    for record in records:
        assert record["cuda_peak_allocated_bytes"] > 0
        assert record["cuda_peak_reserved_bytes"] >= record["cuda_peak_allocated_bytes"]
