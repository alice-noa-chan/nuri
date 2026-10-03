"""Independent regressions for checkpoint continuation and training contracts."""

from __future__ import annotations

import importlib.util
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nuri.checkpoint import load_pretrained, load_training_state, save_pretrained, save_training_state
from nuri.config import tiny_config
from nuri.data import TokenCorpus, build_instruction_batch, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.schedule import WSDSchedule
from nuri.trainer import TrainSettings, canvas_loss, make_optimizer, mlm_loss, train
import nuri.trainer as trainer_module


def _settings(folder: Path) -> TrainSettings:
    return TrainSettings(str(folder), total_steps=10, warmup_steps=1, mix={"mlm": 1.0})


def _fixed_mlm_batch() -> dict:
    return {
        "input_ids": torch.tensor([[3, 1, 5], [6, 7, 1]]),
        "input_valid": torch.ones(2, 3, dtype=torch.bool),
        "labels": torch.tensor([[3, 4, 5], [6, 7, 8]]),
        "target_mask": torch.tensor([[False, True, False], [False, False, True]]),
    }


def _mlm_update(model: NuriModel, optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    mlm_loss(model, _fixed_mlm_batch())["loss"].backward()
    optimizer.step()


def test_checkpoint_restores_the_next_adam_update_exactly(tmp_path):
    """Restored moments must reproduce an uninterrupted update on the same batch."""
    torch.manual_seed(9)
    model = NuriModel(tiny_config())
    optimizer = make_optimizer(model, _settings(tmp_path))
    _mlm_update(model, optimizer)
    path = tmp_path / "state.pt"
    save_training_state(path, model, optimizer, step=1)

    fresh = NuriModel(tiny_config())
    fresh_optimizer = make_optimizer(fresh, _settings(tmp_path))
    load_training_state(path, fresh, fresh_optimizer)
    _mlm_update(model, optimizer)
    _mlm_update(fresh, fresh_optimizer)

    for name, expected in model.state_dict().items():
        assert torch.equal(expected, fresh.state_dict()[name]), name


def test_checkpoint_restores_python_rng(tmp_path):
    """An RNG checkpoint should reproduce the next Python random draw."""
    original_rng = random.getstate()
    try:
        model = NuriModel(tiny_config())
        optimizer = make_optimizer(model, _settings(tmp_path))
        random.seed(5)
        path = tmp_path / "state.pt"
        save_training_state(path, model, optimizer, step=1)
        expected = random.random()
        random.random()
        load_training_state(path, model, optimizer)
        assert random.random() == expected
    finally:
        random.setstate(original_rng)


def test_checkpoint_rejects_a_different_semantic_config(tmp_path):
    """Equal tensor shapes do not make a different RoPE configuration compatible."""
    model = NuriModel(tiny_config(rope_theta=10000.0))
    optimizer = make_optimizer(model, _settings(tmp_path))
    path = tmp_path / "state.pt"
    save_training_state(path, model, optimizer, step=1)
    fresh = NuriModel(tiny_config(rope_theta=100.0))
    fresh_optimizer = make_optimizer(fresh, _settings(tmp_path))
    with pytest.raises(ValueError, match="(?i)config"):
        load_training_state(path, fresh, fresh_optimizer)


def test_default_bf16_export_preserves_ties_and_stage_metadata(tmp_path):
    model = NuriModel(tiny_config(length_head_trained=True))
    expected = model.embedding.weight.detach().to(torch.bfloat16)
    save_pretrained(model, tmp_path)
    restored = load_pretrained(tmp_path, dtype=torch.bfloat16)
    assert restored.embedding.weight.dtype == torch.bfloat16
    assert torch.equal(restored.embedding.weight, expected)
    assert restored.config.length_head_trained
    state = restored.state_dict()
    shared_names = [name for name, tensor in state.items()
                    if tensor.data_ptr() == restored.embedding.weight.data_ptr()]
    assert shared_names == ["embedding.weight"]

    # One change must affect both token input and vocabulary output immediately.
    with torch.no_grad():
        restored.embedding.weight[3, 0] = 0.5
    basis = torch.zeros(1, restored.config.hidden_size, dtype=torch.bfloat16)
    basis[0, 0] = 1
    assert restored.embedding(torch.tensor([3]))[0, 0] == 0.5
    assert restored.vocab_logits(basis)[0, 3] == 0.5


def test_canvas_training_keeps_gradients_through_both_memory_caches():
    """The loss must train the source encoder, recurrent core, and canvas readout."""
    torch.manual_seed(17)
    config = tiny_config()
    model = NuriModel(config)
    batch = build_instruction_batch(
        [([3, 4], [8, 9]), ([5, 6, 7], [10])],
        config,
        generator=torch.Generator().manual_seed(3),
    )
    canvas_loss(model, batch, depth=3, length_weight=0.1)["loss"].backward()
    parameters = {
        "encoder attention": model.encoder.blocks[0].attn.q_proj.weight,
        "core source keys": model.core.cross_attn.k_proj.weight,
        "core source values": model.core.cross_attn.v_proj.weight,
        "core gates": model.core.gate_logits,
        "readout memory keys": model.readout.blocks[0].cross_attn.k_proj.weight,
        "readout memory values": model.readout.blocks[0].cross_attn.v_proj.weight,
        "length head": model.length_head.weight,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_train_script_seeds_before_scratch_initialization(monkeypatch):
    """Identical CLI seeds must not depend on the ambient process RNG state."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("nuri_training_seed_regression", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    captured = []

    monkeypatch.setattr(script, "TokenCorpus", lambda _: object())
    monkeypatch.setattr(script, "base_stage", lambda *args: ({}, {"mlm": 1.0}))

    def capture_initialization(model, *args, **kwargs):
        captured.append(model.embedding.weight.detach().clone())
        return [{"loss": 0.0}]

    monkeypatch.setattr(script, "train", capture_initialization)
    with torch.random.fork_rng():
        for ambient_seed in (41, 67):
            torch.manual_seed(ambient_seed)
            script.main([
                "--stage", "base", "--preset", "tiny", "--corpus", "unused",
                "--out", "unused", "--steps", "1", "--seed", "11",
            ])
    assert torch.equal(captured[0], captured[1])


@pytest.mark.parametrize("total_steps,warmup_steps,decay_fraction", [(10, 20, 0.15), (10, 3, 0.9)])
def test_wsd_rejects_warmup_that_overlaps_decay(total_steps, warmup_steps, decay_fraction):
    """A pre-decay checkpoint must follow warmup instead of being saved during it."""
    with pytest.raises(ValueError):
        WSDSchedule(1.0, total_steps, warmup_steps, decay_fraction)


class PlannedInterruption(RuntimeError):
    """Stop after a real checkpoint without changing the intended LR schedule."""


@pytest.fixture
def middle_checkpoint(tmp_path, monkeypatch):
    """Build a genuine step-3 checkpoint from a run planned for eight steps."""
    corpus_folder = tmp_path / "corpus"
    documents = [[3 + (row + position) % 90 for position in range(24 + row % 12)]
                 for row in range(20)]
    write_corpus(corpus_folder, documents)
    corpus = TokenCorpus(corpus_folder)
    config = tiny_config()
    torch.manual_seed(19)
    model = NuriModel(config)
    tasks, mix = base_stage(corpus, config, batch_size=3, seed=11)
    assert tasks["mlm"].rng is tasks["span"].rng
    assert tasks["mlm"].generator is tasks["span"].generator
    settings = TrainSettings(
        out_dir=str(tmp_path / "interrupted"), total_steps=8, warmup_steps=1,
        peak_lr=2e-3, save_every=3, log_every=1, seed=11, mix=mix,
    )
    original_save = trainer_module._maybe_save

    def save_then_interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[4] == 2:
            raise PlannedInterruption("saved three completed steps")

    with monkeypatch.context() as context:
        context.setattr(trainer_module, "_maybe_save", save_then_interrupt)
        with pytest.raises(PlannedInterruption):
            train(model, tasks, settings, device=torch.device("cpu"))
    checkpoint = Path(settings.out_dir) / "training_state" / "step-3.pt"
    assert checkpoint.is_file()
    return corpus_folder, corpus, config, settings, checkpoint


def _assert_state_equal(expected, actual, location="state"):
    """Compare optimizer trees without weakening exact tensor equality."""
    if torch.is_tensor(expected):
        assert torch.equal(expected, actual), location
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), location
        for key in expected:
            _assert_state_equal(expected[key], actual[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), location
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_state_equal(left, right, f"{location}[{index}]")
    else:
        assert expected == actual, location


def test_actual_interruption_and_resume_match_the_complete_training_trace(middle_checkpoint, tmp_path):
    """A fresh model and aliased task RNGs must continue at the exact next step."""
    _, corpus, config, settings, checkpoint = middle_checkpoint
    torch.manual_seed(19)
    uninterrupted = NuriModel(config)
    full_tasks, _ = base_stage(corpus, config, batch_size=3, seed=11)
    full_settings = replace(settings, out_dir=str(tmp_path / "uninterrupted"))
    full_records = train(uninterrupted, full_tasks, full_settings, device=torch.device("cpu"))

    torch.manual_seed(99)
    resumed = NuriModel(config)
    new_tasks, _ = base_stage(corpus, config, batch_size=3, seed=11)
    assert new_tasks["mlm"].rng is new_tasks["span"].rng
    assert new_tasks["mlm"].generator is new_tasks["span"].generator
    resumed_settings = replace(settings, out_dir=str(tmp_path / "resumed"))
    resumed_records = train(resumed, new_tasks, resumed_settings, device=torch.device("cpu"),
                            resume_from=checkpoint)

    interrupted_records = [json.loads(line) for line in
                           (Path(settings.out_dir) / "train_log.jsonl").read_text().splitlines()]
    assert [record["step"] for record in interrupted_records] == [0, 1, 2]
    assert [record["step"] for record in resumed_records] == list(range(3, 8))
    assert {record["task"] for record in full_records} == {"mlm", "span"}
    for expected, actual in zip(full_records, interrupted_records + resumed_records):
        assert {key: value for key, value in expected.items() if key != "seconds"} == {
            key: value for key, value in actual.items() if key != "seconds"
        }
    _assert_state_equal(uninterrupted.state_dict(), resumed.state_dict(), "model")
    full_state = torch.load(Path(full_settings.out_dir) / "training_state" / "final.pt", weights_only=False)
    resumed_state = torch.load(Path(resumed_settings.out_dir) / "training_state" / "final.pt", weights_only=False)
    _assert_state_equal(full_state["optimizer"], resumed_state["optimizer"], "optimizer")
    _assert_state_equal(full_state["extra"]["totals"], resumed_state["extra"]["totals"], "totals")


@pytest.mark.parametrize("change", [
    {"total_steps": 9}, {"warmup_steps": 2}, {"peak_lr": 4e-3},
])
def test_resume_rejects_silent_schedule_changes(middle_checkpoint, tmp_path, change):
    """Same-run continuation must not silently reinterpret an old schedule."""
    _, corpus, config, settings, checkpoint = middle_checkpoint
    model = NuriModel(config)
    tasks, _ = base_stage(corpus, config, batch_size=3, seed=11)
    altered = replace(settings, out_dir=str(tmp_path / "different-settings"), **change)
    with pytest.raises(ValueError, match="(?i)(settings|schedule|resume)"):
        train(model, tasks, altered, device=torch.device("cpu"), resume_from=checkpoint)


def test_resume_cli_builds_the_model_from_saved_config(middle_checkpoint, monkeypatch, tmp_path):
    """A private checkpoint supplies model dimensions without guessing a preset."""
    corpus_folder, _, config, _, checkpoint = middle_checkpoint
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("nuri_cli_resume_regression", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.delenv("NURI_TOKENIZER", raising=False)

    def reject_guessed_preset(**kwargs):
        raise AssertionError("resume must reconstruct the saved model config")

    def capture_bootstrap(model, tasks, settings, **kwargs):
        assert model.config.to_dict() == config.to_dict()
        assert Path(kwargs["resume_from"]) == checkpoint
        return [{"loss": 0.0}]

    monkeypatch.setattr(script, "PRESETS", {"release-dense": reject_guessed_preset})
    monkeypatch.setattr(script, "train", capture_bootstrap)
    script.main([
        "--stage", "base", "--resume", str(checkpoint), "--corpus", str(corpus_folder),
        "--out", str(tmp_path / "cli-resumed"), "--steps", "8", "--warmup", "1",
        "--seed", "11", "--batch-size", "3", "--lr", "0.002",
    ])
