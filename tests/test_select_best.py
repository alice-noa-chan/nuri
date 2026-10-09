"""Best-validation checkpoint selection: kept by training, released by run_release."""

import json

import pytest
import torch

from nuri import trainer
from nuri.config import tiny_config
from nuri.model import NuriModel
from scripts import train as cli
from test_holdout_cli import args, inputs
from test_run_release import load_script


def report(losses: dict) -> dict:
    return {
        "tasks": {name: {"metrics": {"loss": loss}} for name, loss in losses.items()}
    }


def write_receipt(out, step, losses):
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    (out / "holdout" / f"valid-step{step}.json").write_text(json.dumps(report(losses)))


def test_score_is_the_mean_of_aggregate_objectives_only():
    losses = {"instruction": 2.0, "span": 4.0, "ko/instruction": 100.0}
    assert trainer.holdout_score(report(losses)) == 3.0


def test_update_keeps_only_the_lowest_score_and_ties_keep_the_earlier_step(tmp_path):
    torch.manual_seed(0)
    model = NuriModel(tiny_config())
    settings = trainer.TrainSettings(str(tmp_path), 8, select_best=True)
    for step, loss in ((2, 3.0), (4, 2.0), (6, 2.5), (8, 2.0)):
        write_receipt(tmp_path, step, {"instruction": loss})
        trainer._update_best(model, settings, step - 1, {"tokens": step * 10}, tmp_path)
    best = json.loads((tmp_path / "best.json").read_text())
    assert (best["step"], best["score"], best["tokens"]) == (4, 2.0, 40)
    assert [p.name for p in tmp_path.glob("best-step*")] == ["best-step4"]
    assert (tmp_path / "best-step4" / "model.safetensors").is_file()


def test_folder_left_by_a_crash_is_replaced_and_strays_are_removed(tmp_path):
    model = NuriModel(tiny_config())
    settings = trainer.TrainSettings(str(tmp_path), 8, select_best=True)
    (tmp_path / "best-step2").mkdir()
    (tmp_path / "best-step2" / "partial").write_text("half written")
    (tmp_path / "best-step9").mkdir()  # never named by best.json
    write_receipt(tmp_path, 2, {"instruction": 1.0})
    trainer._update_best(model, settings, 1, {"tokens": 5}, tmp_path)
    assert not (tmp_path / "best-step2" / "partial").exists()
    assert [p.name for p in tmp_path.glob("best-step*")] == ["best-step2"]


def test_training_cli_keeps_the_best_validated_step(tmp_path):
    inputs(tmp_path)
    cli.main(args(tmp_path, extra=["--select-best"]))
    out = tmp_path / "base"
    valid = [
        json.loads(line)
        for line in (out / "holdout_log.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "valid"
    ]
    scores = {entry["step"]: trainer.holdout_score(entry) for entry in valid}
    best = json.loads((out / "best.json").read_text())
    assert best["score"] == min(scores.values())
    assert best["step"] == min(scores, key=lambda step: (scores[step], step))
    assert [p.name for p in out.glob("best-step*")] == [best["folder"]]
    assert (out / best["folder"] / "model.safetensors").is_file()
    assert (out / "final" / "model.safetensors").is_file()  # final is still written


def test_select_best_needs_a_holdout_config():
    with pytest.raises(SystemExit) as error:
        cli.parse_args(
            ["--stage", "it", "--out", "unused", "--steps", "1", "--select-best"]
        )
    assert error.value.code == 2
    runner = load_script()
    plan = {"stages": {"it": {"stage": "it", "steps": 4, "select_best": True}}}
    with pytest.raises(ValueError, match="select_best requires"):
        runner.validate_holdout_settings(plan)


def test_runner_evaluates_and_exports_the_selected_checkpoint(tmp_path):
    runner = load_script()
    stage = tmp_path / "runs" / "it"
    for folder in ("final", "best-step6"):
        (stage / folder).mkdir(parents=True)
        (stage / folder / "model.safetensors").write_bytes(folder.encode())
    (stage / "train_log.jsonl").write_text(json.dumps({"tokens": 900}) + "\n")
    spec = {"stage": "it", "release_name": "nuri-base-it", "evaluate": ["curves"]}
    plan = {"stages": {"it": spec}}
    out = tmp_path / "runs"

    # Without select_best, or before training wrote best.json, final is released.
    assert runner.release_checkpoint(stage, spec) == stage / "final"
    (stage / "best.json").write_text(json.dumps(
        {"folder": "best-step6", "step": 6, "score": 1.5, "tokens": 600}))  # fmt: skip
    assert runner.release_checkpoint(stage, spec) == stage / "final"

    spec["select_best"] = True
    assert runner.release_checkpoint(stage, spec) == stage / "best-step6"
    evaluate = runner.evaluate_command("it", spec, plan, out)
    assert evaluate[evaluate.index("--checkpoint") + 1] == str(stage / "best-step6")
    export = runner.export_command("it", spec, plan, out, 1.0)
    assert export[export.index("--checkpoint") + 1] == str(stage / "best-step6")
    assert export[export.index("--tokens") + 1] == "600"  # tokens at the kept step
    assert "--select-best" in runner.train_command("it", spec, plan, out)

    (stage / "best-step6" / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing checkpoint"):
        runner.release_checkpoint(stage, spec)


def test_patience_counts_validations_since_the_best_step(tmp_path):
    settings = trainer.TrainSettings(
        str(tmp_path), 40, select_best=True, patience=2, valid_every=4
    )
    assert not trainer._patience_spent(settings, 7, tmp_path)  # no best.json yet
    (tmp_path / "best.json").write_text(json.dumps({"step": 8}))
    assert not trainer._patience_spent(settings, 11, tmp_path)  # one validation later
    assert trainer._patience_spent(settings, 15, tmp_path)  # two validations later
    off = trainer.TrainSettings(str(tmp_path), 40, select_best=True, valid_every=4)
    assert not trainer._patience_spent(off, 39, tmp_path)


def test_training_cli_stops_early_and_writes_a_complete_final(tmp_path, monkeypatch):
    # Every validation is worse than the first, so patience 1 ends the run at
    # the second validation (step 4 of 12, validations every 2 steps).
    monkeypatch.setattr(trainer, "holdout_score", lambda report: report["step"])
    inputs(tmp_path)
    run = args(tmp_path, extra=["--select-best", "--patience", "1"])
    run[run.index("--steps") + 1] = "12"
    cli.main(run)
    out = tmp_path / "base"
    stop = json.loads((out / "early_stop.json").read_text())
    assert stop == {
        "stopped_after_step": 4,
        "planned_steps": 12,
        "best_step": 2,
        "patience_validations": 1,
        "valid_every": 2,
    }
    splits = [
        (entry["split"], entry["step"])
        for entry in map(
            json.loads, (out / "holdout_log.jsonl").read_text().splitlines()
        )
    ]
    assert splits == [("valid", 2), ("valid", 4), ("test", 4)]
    assert (out / "final" / "model.safetensors").is_file()
    assert (
        out / "training_state" / "final.pt"
    ).is_file()  # run_release sees it as done
    assert json.loads((out / "best.json").read_text())["step"] == 2
    assert load_script()._finished(out)


def test_patience_needs_select_best():
    with pytest.raises(SystemExit) as error:
        cli.parse_args(
            ["--stage", "it", "--out", "x", "--steps", "1", "--holdout-config", "h.json",
             "--patience", "2"]
        )  # fmt: skip
    assert error.value.code == 2
    plan = {"stages": {"it": {"stage": "it", "steps": 4, "holdout_config": "h.json",
                              "patience": 2}}}  # fmt: skip
    with pytest.raises(ValueError, match="needs select_best"):
        load_script().validate_holdout_settings(plan)
