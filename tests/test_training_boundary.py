"""An exact repair gate interrupts a full schedule without consulting holdouts."""

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri import trainer
from nuri.config import MoEConfig, tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.schedule import WSDSchedule
from nuri.trainer import HoldoutSet, TrainSettings, TrainingInterrupted, train
from scripts import train as cli


def setup_run(tmp_path, name, moe=False):
    folder = tmp_path / "corpus"
    if not folder.exists():
        write_corpus(folder, [[3 + (row + position) % 60 for position in range(14)] for row in range(12)])
    sparse = MoEConfig(num_experts=4, expert_ffn_size=8, top_k=2,
                      input_layers=(1,), readout_layers=(1,)) if moe else None
    config = tiny_config(hidden_size=16, num_heads=2, ffn_size=32, input_layers=1,
                         readout_layers=1, embed_dim=8, moe=sparse)
    torch.manual_seed(17)
    model = NuriModel(config)
    tasks, mix = base_stage(TokenCorpus(folder), config, batch_size=3, seed=9)
    valid_tasks, _ = base_stage(TokenCorpus(folder), config, batch_size=2, seed=71)
    test_tasks, _ = base_stage(TokenCorpus(folder), config, batch_size=2, seed=93)
    settings = TrainSettings(str(tmp_path / name), 8, warmup_steps=2, decay_fraction=0.5,
                             log_every=1, save_every=4, mix=mix, progress=False,
                             embed_dims=(4, 8), grad_accum=2, valid_every=4)
    return model, tasks, settings, HoldoutSet(valid_tasks, 1), HoldoutSet(test_tasks, 1)


def assert_same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("moe", [False, True])
def test_gate_then_resume_is_exact_full_schedule_and_skips_due_assessment(tmp_path, moe):
    model, tasks, settings, valid, test = setup_run(tmp_path, "plain", moe)
    train(model, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    reference = torch.load(tmp_path / "plain/training_state/final.pt", weights_only=False)

    gated, tasks, settings, valid, test = setup_run(tmp_path, "gated", moe)
    settings.stop_after_completed_steps = 4
    with pytest.raises(TrainingInterrupted) as stopped:
        train(gated, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    assert stopped.value.step == 4 and stopped.value.reason == "step_boundary"
    state = torch.load(stopped.value.path, weights_only=False)
    assert state["step"] == 4 and state["extra"]["settings"]["total_steps"] == 8
    assert "stop_after_completed_steps" not in state["extra"]["settings"]
    out = tmp_path / "gated"
    assert not (out / "holdout_log.jsonl").exists()
    assert not (out / "step-4").exists()
    assert not (out / "pre-decay").exists()
    assert not (out / "final").exists()
    assert not (out / "training_state/final.pt").exists()
    schedule = WSDSchedule(settings.peak_lr, 8, 2, 0.5)
    assert state["optimizer"]["param_groups"][0]["lr"] == schedule.lr(3)

    resumed, tasks, settings, valid, test = setup_run(tmp_path, "gated", moe)
    train(resumed, tasks, settings, torch.device("cpu"), resume_from=stopped.value.path,
          validation=valid, test=test)  # boundary omitted
    final = torch.load(out / "training_state/final.pt", weights_only=False)
    for key in ("model", "optimizer", "step", "config", "extra"):
        assert_same(reference[key], final[key])
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == list(range(8))
    assert [record["lr"] for record in records] == [schedule.lr(step) for step in range(8)]
    reports = [json.loads(line) for line in (out / "holdout_log.jsonl").read_text().splitlines()]
    assert [(report["split"], report["step"]) for report in reports] == [("valid", 8), ("test", 8)]
    assert (out / "step-4/model.safetensors").exists()
    assert (out / "pre-decay/model.safetensors").exists()
    assert (out / "final/model.safetensors").exists()


@pytest.mark.parametrize("boundary", [3, 4])
def test_repeating_reached_gate_does_not_advance_or_promote(tmp_path, monkeypatch, boundary):
    model, tasks, settings, valid, test = setup_run(tmp_path, "first")
    settings.stop_after_completed_steps = 4
    with pytest.raises(TrainingInterrupted) as stopped:
        train(model, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    before = torch.load(stopped.value.path, weights_only=False)
    model, tasks, settings, valid, test = setup_run(tmp_path, "again")
    settings.stop_after_completed_steps = boundary
    for task in tasks.values():
        task.next_batch = lambda: pytest.fail("advanced past an already reached gate")
    monkeypatch.setattr(trainer, "_complete_checkpoints", lambda *a, **k: pytest.fail("promoted gate checkpoint"))
    monkeypatch.setattr(trainer, "evaluate_holdout", lambda *a, **k: pytest.fail("consulted holdout at gate"))
    with pytest.raises(TrainingInterrupted) as repeated:
        train(model, tasks, settings, torch.device("cpu"), resume_from=stopped.value.path,
              validation=valid, test=test)
    assert repeated.value.step == 4 and repeated.value.reason == "step_boundary"
    after = torch.load(repeated.value.path, weights_only=False)
    assert_same(before, after)
    assert not (tmp_path / "again/final").exists()


def test_completed_resume_with_gate_does_not_assess_or_create_final_marker(tmp_path, monkeypatch):
    model, tasks, settings, valid, test = setup_run(tmp_path, "complete")
    train(model, tasks, settings, torch.device("cpu"), validation=valid, test=test)
    model, tasks, settings, valid, test = setup_run(tmp_path, "gated-complete")
    settings.stop_after_completed_steps = 4
    monkeypatch.setattr(trainer, "evaluate_holdout", lambda *a, **k: pytest.fail("reconsulted completed test"))
    monkeypatch.setattr(trainer, "_complete_checkpoints", lambda *a, **k: pytest.fail("published completed state"))
    with pytest.raises(TrainingInterrupted) as stopped:
        train(model, tasks, settings, torch.device("cpu"),
              resume_from=tmp_path / "complete/training_state/final.pt", validation=valid, test=test)
    assert stopped.value.step == 8 and stopped.value.reason == "step_boundary"
    assert torch.load(stopped.value.path, weights_only=False)["step"] == 8
    assert not (tmp_path / "gated-complete/final").exists()


def test_first_update_gate_without_holdouts_saves_all_accumulated_work(tmp_path):
    model, tasks, settings, _, _ = setup_run(tmp_path, "first-update")
    settings.valid_every = 0
    settings.stop_after_completed_steps = 1
    with pytest.raises(TrainingInterrupted) as stopped:
        train(model, tasks, settings, torch.device("cpu"))
    state = torch.load(stopped.value.path, weights_only=False)
    assert state["step"] == 1 and state["extra"]["settings"]["grad_accum"] == 2
    assert state["extra"]["totals"]["tokens"] > 0
    assert all(int(moment["step"]) == 1 for moment in state["optimizer"]["state"].values())
    assert not (tmp_path / "first-update/final").exists()


@pytest.mark.parametrize("boundary", [0, -1, 8, 9, True, 2.5])
def test_invalid_boundary_rejected_before_optimizer(tmp_path, monkeypatch, boundary):
    model, tasks, settings, _, _ = setup_run(tmp_path, "invalid")
    settings.stop_after_completed_steps = boundary
    monkeypatch.setattr(trainer, "make_optimizer", lambda *a: pytest.fail("allocated optimizer"))
    with pytest.raises(ValueError, match="positive integer below total_steps"):
        train(model, tasks, settings, torch.device("cpu"))
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("boundary", ["0", "-1", "3", "4"])
def test_invalid_cli_boundary_rejected_before_allocation(monkeypatch, boundary):
    monkeypatch.setattr(cli, "NuriModel", lambda *a: pytest.fail("allocated model"))
    with pytest.raises(SystemExit) as rejected:
        cli.main(["--stage", "base", "--out", "unused", "--steps", "3",
                  "--stop-after-completed-steps", boundary])
    assert rejected.value.code == 2


def test_real_cli_returns_75_then_completes_without_boundary(tmp_path, capsys):
    words = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"]
    words += [f"word{index}" for index in range(48)]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    for shift, split in enumerate(("train", "valid", "test")):
        write_corpus(tmp_path / split, [[8 + (shift * 11 + row + pos) % 48 for pos in range(12)] for row in range(10)])
    policy = {"schema_version": 1, "valid_every": 2, "batch_size": 2}
    for split in ("valid", "test"):
        policy[split] = {"corpus": [split], "batches_per_task": 1, "seed": 9, "depth": 2}
    (tmp_path / "holdout.json").write_text(json.dumps(policy))
    args = ["--stage", "base", "--preset", "tiny", "--out", str(tmp_path / "run"), "--steps", "3",
            "--batch-size", "2", "--tokenizer", str(tmp_path / "tokenizer.json"),
            "--corpus", str(tmp_path / "train"), "--holdout-config", str(tmp_path / "holdout.json"),
            "--save-every", "2", "--no-progress", "--log-every", "1"]
    with pytest.raises(SystemExit) as stopped:
        cli.main([*args, "--stop-after-completed-steps", "2"])
    assert stopped.value.code == trainer.INTERRUPTED_EXIT_CODE == 75
    assert "completed-step boundary after step 2" in capsys.readouterr().out
    out = tmp_path / "run"
    assert not (out / "holdout_log.jsonl").exists()
    assert not (out / "final").exists()
    cli.main([*args, "--resume", str(out / "training_state/latest.pt")])
    reports = [json.loads(line) for line in (out / "holdout_log.jsonl").read_text().splitlines()]
    assert [(report["split"], report["step"]) for report in reports] == [("valid", 3), ("test", 3)]
    assert (out / "final/model.safetensors").exists()
