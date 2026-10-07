"""Optional kernels are deliberate CLI choices and survive release-plan overrides."""

import json

import pytest
import torch

from scripts import run_release
from scripts import train as cli


@pytest.fixture(autouse=True)
def clear_input_environment(monkeypatch):
    for name in (
        "NURI_TOKENIZER",
        "NURI_HOLDOUT_CONFIG",
        "NURI_RECIPE_CONFIG",
        "NURI_ASSESSMENT_RECIPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)


def common(out="unused"):
    return ["--stage", "base", "--preset", "tiny", "--steps", "2", "--out", str(out)]


def test_cli_defaults_preserve_original_execution():
    args = cli.parse_args(common())
    assert args.attention_backend == "sdpa"
    assert not args.liger_kernels
    assert args.vocab_loss_backend == "torch" and args.vocab_chunk_size == 256


@pytest.mark.parametrize(
    "arguments",
    [
        ["--attention-backend", "unknown"],
        ["--vocab-loss-backend", "unknown"],
        ["--vocab-chunk-size", "0"],
        ["--vocab-chunk-size", "-2"],
    ],
)
def test_invalid_cli_options_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(common() + arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--attention-backend", "flash2"],
        ["--attention-backend", "flash3"],
        ["--vocab-loss-backend", "liger"],
        ["--liger-kernels"],
    ],
)
def test_cuda_requests_on_cpu_fail_before_private_inputs(
    tmp_path, monkeypatch, arguments
):
    monkeypatch.setattr(cli, "default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        cli, "prepare_recipe", lambda *a: pytest.fail("private inputs read")
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        cli.main(common(out) + arguments)
    assert not out.exists()


def test_cli_forwards_all_kernel_options_to_training(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "validate_attention_backend", lambda *a: None)
    monkeypatch.setattr(cli, "validate_vocab_loss", lambda *a: None)
    monkeypatch.setattr(cli, "validate_liger", lambda *a: None)
    monkeypatch.setattr(cli, "open_corpus", lambda *a: object())
    monkeypatch.setattr(cli, "base_stage", lambda *a, **k: ({}, {"mlm": 1.0}))
    monkeypatch.setattr(
        cli,
        "train",
        lambda model, tasks, settings, **kwargs: captured.append(settings) or [],
    )
    cli.main(
        common(tmp_path / "out")
        + [
            "--corpus",
            "unused",
            "--no-progress",
            "--attention-backend",
            "flash2",
            "--liger-kernels",
            "--vocab-loss-backend",
            "liger",
            "--vocab-chunk-size",
            "128",
        ]
    )
    assert captured[0].attention_backend == "flash2" and captured[0].liger_kernels
    assert (
        captured[0].vocab_loss_backend == "liger"
        and captured[0].vocab_chunk_size == 128
    )


def test_plan_global_choices_and_stage_overrides(tmp_path):
    plan = {
        "train": {
            "attention_backend": "flash2",
            "liger_kernels": True,
            "vocab_loss_backend": "liger",
            "vocab_chunk_size": 128,
        },
        "stages": {
            "base": {"stage": "base"},
            "it": {
                "stage": "it",
                "attention_backend": "sdpa",
                "liger_kernels": False,
                "vocab_loss_backend": "chunked",
                "vocab_chunk_size": 64,
            },
        },
    }
    run_release.validate_kernel_settings(plan)
    base = run_release.train_command("base", plan["stages"]["base"], plan, tmp_path)
    tuned = run_release.train_command("it", plan["stages"]["it"], plan, tmp_path)
    assert "--liger-kernels" in base and "--liger-kernels" not in tuned
    for command, values in (
        (base, ("flash2", "liger", "128")),
        (tuned, ("sdpa", "chunked", "64")),
    ):
        for flag, expected in zip(run_release.KERNEL_FLAGS, values):
            option = "--" + flag.replace("_", "-")
            assert (
                command.count(option) == 1
                and command[command.index(option) + 1] == expected
            )


@pytest.mark.parametrize(
    "options",
    [
        {"attention_backend": "unknown"},
        {"vocab_loss_backend": "unknown"},
        {"vocab_chunk_size": 0},
        {"vocab_chunk_size": True},
        {"vocab_chunk_size": 1.5},
        {"liger_kernels": "false"},
    ],
)
@pytest.mark.parametrize("global_setting", [True, False])
def test_malformed_plan_fails_before_output(tmp_path, options, global_setting):
    out = tmp_path / "out"
    plan = {
        "out": str(out),
        "train": options if global_setting else {},
        "stages": {"base": {"stage": "base", **({} if global_setting else options)}},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        run_release.main([str(path), "--dry-run"])
    assert not out.exists()
