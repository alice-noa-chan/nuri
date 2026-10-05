import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from nuri.data import write_corpus  # noqa: E402

WORDS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"] + \
        [f"w{i}" for i in range(120)]


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_release.py"
    spec = importlib.util.spec_from_file_location("run_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_plan(tmp_path: Path) -> Path:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (tmp_path / "tok").mkdir()
    tok.save(str(tmp_path / "tok" / "tokenizer.json"))
    write_corpus(tmp_path / "corpus", [[8 + (i + j) % 100 for j in range(30)] for i in range(40)])
    (tmp_path / "it.jsonl").write_text("\n".join(json.dumps(
        {"messages": [{"role": "user", "content": f"w{i} w{i + 1}"}], "response": f"w{i + 2}"}) for i in range(20)),
        encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(
        {"query": f"w{i} w{i + 1}", "positive": f"w{i + 1} w{i}"}) for i in range(12)), encoding="utf-8")
    plan = {
        "out": str(tmp_path / "runs"),
        "data": {"tokenizer": str(tmp_path / "tok"), "corpus": str(tmp_path / "corpus"),
                 "eval_corpus": str(tmp_path / "corpus"), "instructions": str(tmp_path / "it.jsonl"),
                 "pairs": str(tmp_path / "pairs.jsonl"), "eval_pairs": str(tmp_path / "pairs.jsonl")},
        "release": {"gpu": "CPU"},
        "stages": {
            "base": {"stage": "base", "preset": "tiny", "steps": 4, "batch_size": 4, "save_every": 2,
                     "evaluate": ["mlm"], "release_name": "tiny-base"},
            "it": {"stage": "it", "init": "base", "steps": 3, "batch_size": 4, "reasoning_examples": 0,
                   "release_name": "tiny-it"},
            "embed": {"stage": "embed", "init": "base", "steps": 3, "batch_size": 4,
                      "evaluate": ["retrieval"], "release_name": "tiny-embed"},
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_full_plan_runs_skips_and_resumes(tmp_path, capsys):
    plan = make_plan(tmp_path)
    runner = load_script()
    runner.main([str(plan)])
    runs = tmp_path / "runs"
    for name in ("tiny-base", "tiny-it", "tiny-embed"):
        assert (runs / "release" / name / "model.safetensors").exists()
        assert json.loads((runs / "release" / f"{name}.verify.json").read_text())["ok"]
    assert "mlm" in json.loads((runs / "eval" / "base.json").read_text(encoding="utf-8"))
    summary = json.loads((runs / "run_summary.json").read_text())
    assert set(summary) == {"base", "it", "embed"} and summary["base"]["train_seconds"] > 0

    # A second run skips every finished stage.
    capsys.readouterr()
    runner.main([str(plan), "--only", "base", "it"])
    out = capsys.readouterr().out
    assert "[base] already trained" in out and "[it] already trained" in out

    # An interrupted stage resumes from its newest training state.
    shutil.rmtree(runs / "base" / "final")
    (runs / "base" / "training_state" / "final.pt").unlink()
    capsys.readouterr()
    runner.main([str(plan), "--only", "base"])
    assert "--resume" in capsys.readouterr().out
    assert (runs / "base" / "final" / "model.safetensors").exists()


def test_dry_run_prints_commands_only(tmp_path, capsys):
    plan = make_plan(tmp_path)
    load_script().main([str(plan), "--dry-run"])
    out = capsys.readouterr().out
    assert out.count("train.py --stage") == 3
    assert not (tmp_path / "runs" / "base").exists()


def test_plan_switches_reach_the_train_command(tmp_path, capsys):
    plan_path = make_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["train"] = {"tensorboard": True, "gradient_checkpointing": True}
    plan["stages"]["base"].update(grad_accum=2, compile=True)
    plan["stages"]["it"]["gradient_checkpointing"] = False  # a stage can turn a plan-wide switch off
    plan_path.write_text(json.dumps(plan))
    load_script().main([str(plan_path), "--dry-run"])
    lines = [line for line in capsys.readouterr().out.splitlines() if "train.py" in line]
    base, it, embed = lines
    assert "--grad-accum 2" in base and "--compile" in base and "--gradient-checkpointing" in base
    assert "--tensorboard" in it and "--gradient-checkpointing" not in it
    assert "--gradient-checkpointing" in embed


def test_stage_holdout_flags_reach_dry_run_without_implicit_sharing(tmp_path, capsys):
    plan_path = make_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    for index, (name, stage) in enumerate(plan["stages"].items()):
        stage["holdout_config"] = str(tmp_path / f"{name}-holdout.json")
        stage["valid_every"] = index
    plan_path.write_text(json.dumps(plan))
    load_script().main([str(plan_path), "--dry-run"])
    lines = [line for line in capsys.readouterr().out.splitlines() if "train.py" in line]
    for index, (name, line) in enumerate(zip(plan["stages"], lines)):
        assert f"--holdout-config {tmp_path / f'{name}-holdout.json'}" in line
        assert f"--valid-every {index}" in line
    assert not (tmp_path / "runs" / "base").exists()


@pytest.mark.parametrize("changes", [
    {"valid_every": 1},
    {"holdout_config": ""},
    {"holdout_config": None},
    {"holdout_config": ["private.json"]},
    {"holdout_config": "private.json", "valid_every": -1},
    {"holdout_config": "private.json", "valid_every": True},
    {"holdout_config": "private.json", "valid_every": 0.5},
])
def test_invalid_holdout_plan_fails_before_any_stage_or_output(tmp_path, monkeypatch, changes):
    plan_path = make_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["stages"]["base"].update(changes)
    plan_path.write_text(json.dumps(plan))
    runner = load_script()
    monkeypatch.setattr(runner, "run", lambda *_: pytest.fail("must validate before launching"))
    with pytest.raises(ValueError, match="holdout_config|valid_every"):
        runner.main([str(plan_path)])
    assert not (tmp_path / "runs").exists()


def test_holdout_plan_requires_stage_values_instead_of_ignored_defaults(tmp_path):
    plan_path = make_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["train"] = {"holdout_config": "private.json", "valid_every": 1}
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="per stage"):
        load_script().main([str(plan_path), "--dry-run"])


def test_actual_release_stage_records_periodic_valid_and_end_only_test(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    plan_path = make_plan(tmp_path)
    write_corpus(tmp_path / "valid-corpus", [[20 + (i + j) % 20 for j in range(20)] for i in range(8)])
    write_corpus(tmp_path / "test-corpus", [[50 + (i + j) % 20 for j in range(20)] for i in range(8)])
    holdout = {
        "schema_version": 1, "valid_every": 0, "batch_size": 2,
        "valid": {"corpus": ["valid-corpus"], "batches_per_task": 1, "seed": 4, "depth": 2},
        "test": {"corpus": ["test-corpus"], "batches_per_task": 1, "seed": 5, "depth": 2},
    }
    holdout_path = tmp_path / "base-holdout.json"
    holdout_path.write_text(json.dumps(holdout))
    plan = json.loads(plan_path.read_text())
    plan["stages"] = {"base": plan["stages"]["base"]}
    plan["stages"]["base"].update(steps=2, holdout_config=str(holdout_path), valid_every=1)
    plan["stages"]["base"].pop("evaluate")
    plan["stages"]["base"].pop("release_name")
    plan_path.write_text(json.dumps(plan))
    runner = load_script()
    runner.main([str(plan_path)])
    path = tmp_path / "runs" / "base" / "holdout_log.jsonl"
    reports = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["split"], r["step"]) for r in reports] == [("valid", 1), ("valid", 2), ("test", 2)]
    assert (tmp_path / "runs" / "base" / "final" / "model.safetensors").exists()
    first_bytes = path.read_bytes()
    runner.main([str(plan_path)])
    assert path.read_bytes() == first_bytes


def finished_evaluation_plan(tmp_path, monkeypatch):
    """Exercise runner receipts without repeating model training in each binding test."""
    plan_path = make_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    stage = plan["stages"]["base"]
    stage.pop("release_name")
    plan["stages"] = {"base": stage}
    plan_path.write_text(json.dumps(plan))
    stage_dir = tmp_path / "runs" / "base"
    (stage_dir / "final").mkdir(parents=True)
    (stage_dir / "training_state").mkdir()
    (stage_dir / "final" / "model.safetensors").write_bytes(b"retained-weight-bytes")
    (stage_dir / "final" / "config.json").write_text("{}")
    (stage_dir / "final" / "tokenizer.json").write_bytes((tmp_path / "tok" / "tokenizer.json").read_bytes())
    (stage_dir / "training_state" / "final.pt").write_bytes(b"finished-state")
    runner = load_script()
    calls = []

    def evaluate(command, dry_run):
        assert not dry_run and "evaluate.py" in command[1]
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"assessment": len(calls)}))
        return 0

    monkeypatch.setattr(runner, "run", evaluate)
    return runner, plan_path, plan, stage_dir, calls


def test_completed_evaluation_reuses_only_verified_receipt(tmp_path, monkeypatch, capsys):
    runner, plan_path, _, _, calls = finished_evaluation_plan(tmp_path, monkeypatch)
    runner.main([str(plan_path)])
    summary_path = tmp_path / "runs" / "run_summary.json"
    summary_bytes = summary_path.read_bytes()
    runner.main([str(plan_path)])
    assert len(calls) == 1
    assert "evaluation already verified" in capsys.readouterr().out
    assert summary_path.read_bytes() == summary_bytes
    receipt = json.loads(summary_bytes)["base"]["evaluation_receipt"]
    assert receipt["returncode"] == 0
    assert set(receipt["binding"]["input_sha256"]) == {"corpus"}
    assert set(receipt["binding"]["environment"]) == {
        "NURI_EVAL_CORPUS", "NURI_EVAL_PAIRS", "NURI_EVAL_INSTRUCTIONS", "CUDA_VISIBLE_DEVICES"}


@pytest.mark.parametrize("changed", ["weights", "config", "tokenizer", "options", "source", "runtime", "input", "manifest"])
def test_changed_evaluation_binding_reruns_and_preserves_prior_evidence(tmp_path, monkeypatch, changed):
    runner, plan_path, plan, stage_dir, calls = finished_evaluation_plan(tmp_path, monkeypatch)
    manifest = tmp_path / "inputs.json"
    manifest.write_text('{"files": []}')
    plan["data"]["eval_input_manifest"] = str(manifest)
    plan_path.write_text(json.dumps(plan))
    runner.main([str(plan_path)])
    result_path = tmp_path / "runs" / "eval" / "base.json"
    previous = result_path.read_bytes()
    if changed in ("weights", "config", "tokenizer"):
        name = {"weights": "model.safetensors", "config": "config.json", "tokenizer": "tokenizer.json"}[changed]
        path = stage_dir / "final" / name
        path.write_bytes(path.read_bytes() + b" ")
    elif changed == "options":
        plan["stages"]["base"]["evaluate"] = ["denoise"]
        plan_path.write_text(json.dumps(plan))
    elif changed == "source":
        monkeypatch.setattr(runner, "evaluation_source_sha256", lambda: "changed-runtime-source")
    elif changed == "runtime":
        monkeypatch.setattr(runner.platform, "platform", lambda: "changed-platform")
    elif changed == "input":
        path = tmp_path / "corpus" / "tokens.bin"
        path.write_bytes(path.read_bytes() + b"\x08\x00\x00\x00")
    elif changed == "manifest":
        manifest.write_text('{"files": [], "revision": 2}')
    runner.main([str(plan_path)])
    assert len(calls) == 2
    history = tmp_path / "runs" / "eval" / "history" / "base"
    copies = list(history.glob("*/base.json"))
    assert len(copies) == 1 and copies[0].read_bytes() == previous
    assert len(list(history.glob("*/receipt.json"))) == 1
    assert result_path.read_bytes() != previous


@pytest.mark.parametrize("damage", ["corrupt", "missing", "no_receipt"])
def test_output_alone_or_damaged_result_never_skips_evaluation(tmp_path, monkeypatch, damage):
    runner, plan_path, _, _, calls = finished_evaluation_plan(tmp_path, monkeypatch)
    runner.main([str(plan_path)])
    output = tmp_path / "runs" / "eval" / "base.json"
    summary_path = tmp_path / "runs" / "run_summary.json"
    if damage == "corrupt":
        output.write_text("not JSON")
    elif damage == "missing":
        output.unlink()
    else:
        summary = json.loads(summary_path.read_text())
        summary["base"].pop("evaluation_receipt")
        summary_path.write_text(json.dumps(summary))
    runner.main([str(plan_path)])
    assert len(calls) == 2
    assert json.loads(output.read_text()) == {"assessment": 2}


def test_success_without_a_fresh_result_cannot_certify_old_output(tmp_path, monkeypatch):
    runner, plan_path, _, _, _ = finished_evaluation_plan(tmp_path, monkeypatch)
    runner.main([str(plan_path)])
    summary_path = tmp_path / "runs" / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["base"].pop("evaluation_receipt")
    summary_path.write_text(json.dumps(summary))
    monkeypatch.setattr(runner, "run", lambda *_: 0)
    with pytest.raises(ValueError, match="valid JSON object"):
        runner.main([str(plan_path)])
    assert "evaluation_receipt" not in json.loads(summary_path.read_text())["base"]
    assert len(list((tmp_path / "runs" / "eval" / "history" / "base").glob("*/base.json"))) == 1


def test_interrupted_evaluation_leaves_no_success_receipt(tmp_path, monkeypatch):
    runner, plan_path, _, _, calls = finished_evaluation_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "run", lambda *_: runner.INTERRUPTED_EXIT_CODE)
    with pytest.raises(SystemExit) as stopped:
        runner.main([str(plan_path)])
    assert stopped.value.code == 75 and not calls
    summary = json.loads((tmp_path / "runs" / "run_summary.json").read_text())
    assert "evaluation_receipt" not in summary["base"]


def test_input_changes_during_evaluation_cannot_receive_receipt(tmp_path, monkeypatch):
    runner, plan_path, _, _, _ = finished_evaluation_plan(tmp_path, monkeypatch)
    normal_run = runner.run

    def mutate_input(command, dry_run):
        code = normal_run(command, dry_run)
        path = tmp_path / "corpus" / "tokens.bin"
        path.write_bytes(path.read_bytes() + b"\x09\x00\x00\x00")
        return code

    monkeypatch.setattr(runner, "run", mutate_input)
    with pytest.raises(ValueError, match="binding changed"):
        runner.main([str(plan_path)])
    summary = json.loads((tmp_path / "runs" / "run_summary.json").read_text())
    assert "evaluation_receipt" not in summary["base"]


@pytest.mark.parametrize("suite,input_key,filename", [
    ("retrieval", "pairs", "pairs.jsonl"),
    ("instructions", "instructions", "it.jsonl"),
])
def test_jsonl_input_bytes_are_bound_only_when_selected(tmp_path, monkeypatch, suite, input_key, filename):
    runner, plan_path, plan, _, calls = finished_evaluation_plan(tmp_path, monkeypatch)
    plan["stages"]["base"]["evaluate"] = [suite]
    plan["data"]["eval_instructions"] = str(tmp_path / "it.jsonl")
    plan_path.write_text(json.dumps(plan))
    runner.main([str(plan_path)])
    summary_path = tmp_path / "runs" / "run_summary.json"
    binding = json.loads(summary_path.read_text())["base"]["evaluation_receipt"]["binding"]
    assert set(binding["input_sha256"]) == {input_key}
    # No corpus suite consumes these bytes, even though the legacy command
    # includes the corpus argument. Unused corpus changes must not reread test.
    (tmp_path / "corpus" / "tokens.bin").write_bytes(b"unused corpus changed")
    runner.main([str(plan_path)])
    assert len(calls) == 1
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b"\n")
    runner.main([str(plan_path)])
    assert len(calls) == 2
