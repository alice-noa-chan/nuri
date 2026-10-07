import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch

from nuri.generation import generate
from nuri.synthetic_generation import MASK_ID, PAD_ID, TaskSpec, make_examples

SPEC = TaskSpec()


def load_script():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "generation_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("generation_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = load_script()


def small_model(run):
    torch.manual_seed(0)
    return script.build_model(run, SPEC).eval()


def batch_for(task, count=6, eos_prob=1.0):
    examples = make_examples(task, count, seed=0, spec=SPEC)
    return script.make_batch(examples, random.Random(0), eos_prob)


def test_experiment_grid_has_unique_keys_and_expected_rows():
    runs = script.experiment_runs(list(script.GROUPS), [0, 1])
    keys = [run.key for run in runs]
    assert len(keys) == len(set(keys))
    rows = {run.row for run in runs}
    assert "tasks/branch-single/mask" in rows
    assert "draft/branch/self-draft" in rows
    assert "eos/count/mask+eos0.1" in rows
    assert "readout/copy/mask+readout4" in rows
    assert "readout/copy/mask+pos-from-start" in rows
    assert "readout/reverse/mask+pos-from-end" in rows
    assert "readout/copy/mask+pos-memory" in rows
    assert len(runs) == 2 * len(rows)


def test_runs_reject_unknown_options():
    with pytest.raises(ValueError):
        script.Run("tasks", "unknown", 0)
    with pytest.raises(ValueError):
        script.Run("tasks", "copy", 0, canvas_training="other")
    with pytest.raises(ValueError):
        script.Run("tasks", "copy", 0, input_positions="middle")


def test_mask_training_matches_the_base_recipe():
    run = script.Run("tasks", "copy", 0)
    model = small_model(run)
    batch = batch_for("copy")
    out = script.corrupt_for_training(
        model, batch, "mask", 2, torch.Generator().manual_seed(0)
    )
    masked = out["loss_mask"]
    assert torch.equal(
        out["canvas"][masked], torch.full_like(out["canvas"][masked], MASK_ID)
    )
    assert torch.equal(out["canvas"][~masked], batch["targets"][~masked])


@pytest.mark.parametrize("training", ["random-noise", "self-draft"])
def test_revision_training_scores_every_position(training):
    run = script.Run("draft", "reverse", 0, canvas_training=training)
    model = small_model(run)
    batch = batch_for("reverse", count=32)
    out = script.corrupt_for_training(
        model, batch, training, 2, torch.Generator().manual_seed(1)
    )
    valid = batch["canvas_valid"]
    assert torch.equal(out["loss_mask"], valid)
    canvas = out["canvas"][valid]
    assert not (canvas == PAD_ID).any()
    changed = (canvas != MASK_ID) & (canvas != batch["targets"][valid])
    assert changed.any(), "some visible tokens must differ from the gold answer"
    assert (out["canvas"][~valid] == PAD_ID).all()


def test_unknown_options_are_rejected():
    run = script.Run("tasks", "copy", 0)
    model = small_model(run)
    with pytest.raises(ValueError):
        script.corrupt_for_training(model, batch_for("copy"), "other", 2, None)
    with pytest.raises(ValueError):
        script.decode(model, make_examples("copy", 2, seed=0, spec=SPEC), "other", 1)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("from-start", [[0, 1, 2, 0, 0], [0, 1, 2, 3, 4]]),
        ("from-end", [[2, 1, 0, 0, 0], [4, 3, 2, 1, 0]]),
    ],
)
def test_input_position_ids(mode, expected):
    run = script.Run("readout", "copy", 0, input_positions=mode)
    model = small_model(run)
    assert isinstance(model, script.InputPositionNuriModel)
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 8, 9]])
    assert model.position_ids(ids != PAD_ID).tolist() == expected
    with pytest.raises(ValueError):
        script.InputPositionNuriModel(model.config, "middle")


def test_end_positions_change_the_memory_of_a_shared_prefix():
    run = script.Run("readout", "reverse", 0, input_positions="from-end")
    model = small_model(run)
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 8, 9]])
    x, valid = model.encode(ids)
    assert valid.tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    # The same tokens at a different distance from the end get different
    # position vectors, so the memory differs even for the shared prefix.
    assert not torch.allclose(x[0, :3], x[1, :3])
    with pytest.raises(ValueError):
        model.encode(ids, segments=torch.zeros_like(ids))


def test_revise_decoder_equals_generate_for_one_round():
    run = script.Run("tasks", "count", 0)
    model = small_model(run)
    examples = make_examples("count", 5, seed=2, spec=SPEC)
    standard = script.decode(model, examples, "standard", 1)
    revised = script.decode(model, examples, "revise", 1)
    assert standard == revised
    for example, canvas in zip(examples, revised):
        assert len(canvas) == len(example.target) + 1
        assert MASK_ID not in canvas and PAD_ID not in canvas


def test_standard_decoder_is_nuri_generate():
    run = script.Run("tasks", "reverse", 0)
    model = small_model(run)
    examples = make_examples("reverse", 4, seed=3, spec=SPEC)
    decoded = script.decode(model, examples, "standard", 4)
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(e.input_ids) for e in examples], batch_first=True
    )
    lengths = torch.tensor([len(e.target) + 1 for e in examples])
    canvas = generate(model, ids, rounds=4, canvas_len=lengths).canvas
    assert decoded == [canvas[i, :n].tolist() for i, n in enumerate(lengths.tolist())]


def test_revise_decoder_follows_the_remask_schedule(monkeypatch):
    run = script.Run("tasks", "copy", 0)
    model = small_model(run)
    seen_masks = []
    original = model.readout_hidden

    def recording(canvas, canvas_valid, cache):
        seen_masks.append(int((canvas == MASK_ID).sum()))
        return original(canvas, canvas_valid, cache)

    monkeypatch.setattr(model, "readout_hidden", recording)
    ids = torch.tensor([[4, 5, 6, 7]])
    script.revise_generate(model, ids, ids != PAD_ID, torch.tensor([8]), rounds=4)
    assert seen_masks == [8, 6, 4, 2]


def test_learning_rate_schedule():
    total = 1000
    assert script.learning_rate_factor(0, total) == pytest.approx(
        1 / script.WARMUP_STEPS
    )
    assert script.learning_rate_factor(script.WARMUP_STEPS, total) == pytest.approx(1.0)
    assert script.learning_rate_factor(total, total) == pytest.approx(0.1)


def test_main_trains_resumes_and_summarizes(tmp_path, capsys):
    argv = [
        "--out",
        str(tmp_path),
        "--groups",
        "eos",
        "--seeds",
        "0",
        "1",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--eval-examples",
        "3",
        "--threads",
        "1",
    ]
    summary = script.main(argv)
    assert {entry["row"] for entry in summary} == {
        "eos/count/mask+eos0.1",
        "eos/branch-single/mask+eos0.1",
    }
    assert all(entry["seeds"] == 2 for entry in summary)
    lines = (tmp_path / "results-0.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    result = json.loads(lines[0])
    assert set(result["scores"]) == {
        "standard_s1",
        "standard_s4",
        "revise_s1",
        "revise_s4",
    }
    assert 0.0 <= result["scores"]["standard_s4"]["token_accuracy"] <= 1.0

    capsys.readouterr()
    script.main(argv)  # every run is already finished, so nothing retrains
    assert "[1/4]" not in capsys.readouterr().out
    assert (
        len((tmp_path / "results-0.jsonl").read_text(encoding="utf-8").splitlines())
        == 4
    )
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(saved) == 2


def test_shards_split_the_runs(tmp_path):
    common = [
        "--out",
        str(tmp_path),
        "--groups",
        "eos",
        "--seeds",
        "0",
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--eval-examples",
        "2",
    ]
    script.main(common + ["--shard", "0", "2"])
    script.main(common + ["--shard", "1", "2"])
    first = (tmp_path / "results-0.jsonl").read_text(encoding="utf-8").splitlines()
    second = (tmp_path / "results-1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first) == len(second) == 1
    assert json.loads(first[0])["key"] != json.loads(second[0])["key"]


def test_invalid_arguments_are_rejected(tmp_path):
    with pytest.raises(SystemExit):
        script.parse_args(["--out", str(tmp_path), "--shard", "2", "2"])
    with pytest.raises(SystemExit):
        script.parse_args(["--out", str(tmp_path), "--steps", "0"])


def test_memory_variant_uses_the_package_option():
    run = script.Run("readout", "copy", 0, input_positions="memory")
    model = small_model(run)
    assert type(model) is script.NuriModel
    assert model.config.memory_position_encoding == "learned"
    assert model.readout.memory_positions is not None
    baseline = small_model(script.Run("readout", "copy", 0))
    assert baseline.config.memory_position_encoding == "none"
