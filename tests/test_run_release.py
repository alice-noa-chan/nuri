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
