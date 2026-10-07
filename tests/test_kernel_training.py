"""Kernel options preserve task objectives, tie weights and bind exact resumes."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from nuri.config import tiny_config
from nuri.layers import Attention
from nuri.model import NuriModel
from nuri.trainer import (
    KERNEL_SETTING_DEFAULTS,
    Task,
    TrainingInterrupted,
    TrainSettings,
    _objective_settings,
    _resume,
    _save_state,
    _task_loss,
    make_optimizer,
    train,
)


def configuration():
    return tiny_config(
        hidden_size=16,
        num_heads=2,
        ffn_size=32,
        input_layers=1,
        readout_layers=1,
        embed_dim=8,
    )


def batch(kind):
    ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 0]])
    valid = ids != 0
    mask = torch.tensor([[True, False, True, False], [True, False, False, False]])
    if kind == "mlm":
        return {
            "input_ids": ids.masked_fill(mask, 1),
            "input_valid": valid,
            "target_mask": mask,
            "labels": ids,
        }
    canvas = torch.tensor([[12, 13, 14, 2], [15, 2, 0, 0]])
    return {
        "input_ids": ids,
        "input_valid": valid,
        "canvas": canvas.masked_fill(mask, 1),
        "canvas_valid": canvas != 0,
        "loss_mask": mask,
        "targets": canvas,
        "lengths": torch.tensor([4, 2]),
    }


@pytest.mark.parametrize("kind", ["mlm", "span", "instruction"])
def test_task_loss_matches_all_parameter_gradients_and_length_objective(kind):
    torch.manual_seed(21)
    reference = NuriModel(configuration())
    model = copy.deepcopy(reference)
    defaults = TrainSettings("", 2)
    chunked = replace(defaults, vocab_loss_backend="chunked", vocab_chunk_size=2)
    expected = _task_loss(reference, kind, batch(kind), 2, defaults)
    actual = _task_loss(model, kind, batch(kind), 2, chunked)
    assert expected.keys() == actual.keys()
    for key in expected:
        if torch.is_tensor(expected[key]):
            torch.testing.assert_close(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]
    (expected["loss"] * 0.37).backward()
    (actual["loss"] * 0.37).backward()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), model.named_parameters()
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, atol=2e-6, rtol=2e-5, msg=name)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"vocab_chunk_size": 0}, "positive integer"),
        ({"vocab_loss_backend": "unknown"}, "backend"),
        ({"vocab_loss_backend": "liger"}, "NVIDIA CUDA"),
        ({"attention_backend": "flash2"}, "NVIDIA CUDA"),
        ({"liger_kernels": True}, "NVIDIA CUDA"),
    ],
)
def test_invalid_options_fail_before_output_or_optimizer(
    tmp_path, monkeypatch, options, match
):
    monkeypatch.setattr(
        "nuri.trainer.make_optimizer", lambda *a: pytest.fail("optimizer created")
    )
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        train(
            NuriModel(configuration()),
            {},
            TrainSettings(str(out), 2, **options),
            device="cpu",
        )
    assert not out.exists()


def test_chunked_training_resume_is_exact(tmp_path):
    torch.manual_seed(22)
    initial = NuriModel(configuration())

    def tasks():
        return {"mlm": Task("mlm", lambda: batch("mlm"))}

    settings = TrainSettings(
        str(tmp_path / "full"),
        4,
        warmup_steps=1,
        mix={"mlm": 1},
        seed=23,
        progress=False,
        log_every=1,
        vocab_loss_backend="chunked",
        vocab_chunk_size=2,
    )
    full = copy.deepcopy(initial)
    train(full, tasks(), settings, device="cpu")
    split = copy.deepcopy(initial)
    partial = replace(
        settings, out_dir=str(tmp_path / "split"), stop_after_completed_steps=2
    )
    with pytest.raises(TrainingInterrupted) as stop:
        train(split, tasks(), partial, device="cpu")
    restored = NuriModel(configuration())
    train(
        restored,
        tasks(),
        replace(settings, out_dir=str(tmp_path / "resumed")),
        device="cpu",
        resume_from=stop.value.path,
    )
    for name, value in full.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name
    full_state = torch.load(
        tmp_path / "full/training_state/final.pt", weights_only=False
    )
    resumed_state = torch.load(
        tmp_path / "resumed/training_state/final.pt", weights_only=False
    )
    for key, state in full_state["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed_state["optimizer"]["state"][key][name])
    saved = json.loads((tmp_path / "resumed/train_settings.json").read_text())
    assert saved["vocab_loss_backend"] == "chunked" and saved["vocab_chunk_size"] == 2
    assert all(
        layer.attention_backend == "sdpa"
        for layer in restored.modules()
        if isinstance(layer, Attention)
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("attention_backend", "flash2"),
        ("liger_kernels", True),
        ("vocab_loss_backend", "chunked"),
        ("vocab_chunk_size", 32),
    ],
)
def test_resume_refuses_each_runtime_change_and_accepts_legacy_defaults(
    tmp_path, key, value
):
    import random

    model = NuriModel(configuration())
    settings = TrainSettings(str(tmp_path), 2)
    optimizer = make_optimizer(model, settings)
    rng = random.Random(1)
    path = _save_state(
        model,
        optimizer,
        settings,
        0,
        {"tokens": 0, "targets": 0},
        tmp_path,
        rng,
        {},
        "state.pt",
    )
    state = torch.load(path, weights_only=False)
    for setting in KERNEL_SETTING_DEFAULTS:
        del state["extra"]["settings"][setting]
    torch.save(state, path)
    assert _resume(path, model, optimizer, rng, {}, settings)[0] == 1
    with pytest.raises(ValueError, match=key):
        _resume(path, model, optimizer, rng, {}, replace(settings, **{key: value}))


def test_assessment_receipt_binds_opt_in_numerical_execution():
    defaults = TrainSettings("", 2)
    assert "runtime_kernels" not in _objective_settings(defaults)
    changed = replace(defaults, vocab_loss_backend="chunked", vocab_chunk_size=2)
    assert _objective_settings(changed)["runtime_kernels"] == {
        **KERNEL_SETTING_DEFAULTS,
        "vocab_loss_backend": "chunked",
        "vocab_chunk_size": 2,
    }
