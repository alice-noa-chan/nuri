"""Independent integration tests for the optional native GQA runtime switch."""

import json
from dataclasses import replace

import pytest
import torch

from nuri.config import tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.layers import Attention
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.trainer import TrainSettings, train
import nuri.trainer as trainer_module
from scripts import run_release, train as train_cli
from tools import benchmark_nuri


def gqa_config(**overrides):
    return tiny_config(hidden_size=16, num_heads=4, num_kv_heads=2, ffn_size=32,
                       input_layers=1, readout_layers=1, embed_dim=8, **overrides)


def attention_layers(model):
    return [module for module in model.modules() if isinstance(module, Attention)]


def assert_state_equal(expected, actual):
    """Compare model/optimizer state trees without relaxing tensor equality."""
    if torch.is_tensor(expected):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_state_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_state_equal(left, right)
    else:
        assert expected == actual


def test_model_runtime_switch_reaches_all_stacks_without_changing_weights_or_config():
    model = NuriModel(gqa_config())
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    config = model.config.to_dict()
    layers = attention_layers(model)
    assert len(layers) == 5  # encoder, two core branches, two readout branches
    assert all(not layer.native_gqa for layer in layers)
    model.set_native_gqa(True)
    assert all(layer.native_gqa for layer in layers)
    model.set_native_gqa(False)
    assert all(not layer.native_gqa for layer in layers)
    assert_state_equal(initial, model.state_dict())
    assert model.config.to_dict() == config


@pytest.mark.parametrize("native,training", [(False, False), (True, False), (True, True)])
def test_benchmark_report_matches_the_actual_grouped_sdpa_execution(monkeypatch, native, training):
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def capture(query, key, value, *args, **kwargs):
        # The capability probe has 2 query heads; model calls have 4.
        if query.shape[1] == 4:
            calls.append((key.shape[1], kwargs.get("enable_gqa", False)))
        return original(query, key, value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    report = benchmark_nuri.run_benchmark(
        gqa_config(), task="denoise" if training else "generate", training=training,
        batch_size=2, input_length=4, output_length=3, steps=2, rounds=2,
        warmup=0, iterations=1, native_gqa=native,
    )
    assert report["native_gqa"] is native
    assert report["attention_execution"] == ("native_sdpa_gqa" if native else "repeat_kv_gqa")
    assert report["latency_seconds"]["mean"] > 0
    assert calls and set(calls) == {(2, True) if native else (4, False)}


def test_mha_benchmark_does_not_report_gqa_execution():
    report = benchmark_nuri.run_benchmark(
        tiny_config(hidden_size=16, num_heads=4, ffn_size=32,
                    input_layers=1, readout_layers=1, embed_dim=8),
        task="embed-fast", input_length=4, warmup=0, iterations=1, native_gqa=True,
    )
    assert report["native_gqa"] is True
    assert report["attention_execution"] == "mha"


def test_benchmark_cli_passes_the_native_flag_to_the_real_runtime(tmp_path, capsys):
    config = tmp_path / "config.json"
    gqa_config().save(config)
    benchmark_nuri.main([
        "--config", str(config), "--native-gqa", "--task", "embed-fast",
        "--input-length", "4", "--warmup", "0", "--iterations", "1", "--device", "cpu",
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["native_gqa"] is True
    assert report["attention_execution"] == "native_sdpa_gqa"


def test_train_cli_and_release_plan_preserve_native_switch_overrides(tmp_path, monkeypatch):
    captured = []

    def capture_training(model, tasks, settings, **kwargs):
        captured.append(settings)
        return []

    monkeypatch.setattr(train_cli, "train", capture_training)
    monkeypatch.setattr(train_cli, "open_corpus", lambda _: object())
    monkeypatch.setattr(train_cli, "base_stage", lambda *a, **k: ({}, {"mlm": 1.0}))
    monkeypatch.delenv("NURI_TOKENIZER", raising=False)
    train_cli.main([
        "--stage", "base", "--preset", "tiny", "--corpus", "unused",
        "--out", str(tmp_path / "cli"), "--steps", "2", "--native-gqa", "--no-progress",
    ])
    assert captured[0].native_gqa is True
    plan = {"train": {"native_gqa": True}, "stages": {"base": {}, "it": {}}}
    enabled = run_release.train_command("base", {"stage": "base"}, plan, tmp_path)
    disabled = run_release.train_command("it", {"stage": "it", "native_gqa": False}, plan, tmp_path)
    assert "--native-gqa" in enabled
    assert "--native-gqa" not in disabled


def test_training_enables_native_gqa_before_compiling_blocks(tmp_path, monkeypatch):
    config = gqa_config()
    model = NuriModel(config)
    corpus_dir = tmp_path / "corpus"
    write_corpus(corpus_dir, [[3 + position % 20 for position in range(20)]] * 4)
    tasks, mix = base_stage(TokenCorpus(corpus_dir), config, batch_size=2, seed=11)
    compiled = []

    def inspect_before_compile(current):
        assert all(layer.native_gqa for layer in attention_layers(current))
        compiled.append(current)

    monkeypatch.setattr(trainer_module, "_compile_blocks", inspect_before_compile)
    settings = TrainSettings(str(tmp_path / "compiled"), total_steps=2, warmup_steps=1,
                             mix=mix, native_gqa=True, compile=True, progress=False)
    train(model, tasks, settings, device=torch.device("cpu"))
    assert compiled == [model]


@pytest.fixture
def interrupted_run(tmp_path, monkeypatch):
    """Use the normal base-stage task RNGs and real checkpoint interruption."""
    corpus_dir = tmp_path / "corpus"
    documents = [[3 + (row + position) % 90 for position in range(20 + row % 4)]
                 for row in range(12)]
    write_corpus(corpus_dir, documents)
    corpus = TokenCorpus(corpus_dir)

    def build(native):
        config = gqa_config()
        torch.manual_seed(19)
        model = NuriModel(config)
        tasks, mix = base_stage(corpus, config, batch_size=2, seed=11)
        settings = TrainSettings(
            str(tmp_path / "interrupted"), total_steps=6, warmup_steps=1, peak_lr=2e-3,
            mix=mix, seed=11, save_every=2, log_every=1, progress=False, native_gqa=native,
        )
        original = trainer_module._maybe_save

        class PlannedInterruption(RuntimeError):
            pass

        def save_then_interrupt(*args, **kwargs):
            original(*args, **kwargs)
            if args[4] == 1:
                raise PlannedInterruption("saved two completed steps")

        with monkeypatch.context() as context:
            context.setattr(trainer_module, "_maybe_save", save_then_interrupt)
            with pytest.raises(PlannedInterruption):
                train(model, tasks, settings, device=torch.device("cpu"))
        checkpoint = tmp_path / "interrupted" / "training_state" / "step-2.pt"
        assert checkpoint.is_file()
        return corpus, config, settings, checkpoint

    return build


def test_native_training_resume_matches_uninterrupted_weights_adam_and_task_rngs(interrupted_run, tmp_path):
    corpus, config, settings, checkpoint = interrupted_run(True)
    torch.manual_seed(19)
    uninterrupted = NuriModel(config)
    tasks, _ = base_stage(corpus, config, batch_size=2, seed=11)
    full_settings = replace(settings, out_dir=str(tmp_path / "full"))
    full_records = train(uninterrupted, tasks, full_settings, device=torch.device("cpu"))
    resumed = NuriModel(config)
    tasks, _ = base_stage(corpus, config, batch_size=2, seed=11)
    assert tasks["mlm"].rng is tasks["span"].rng
    assert tasks["mlm"].generator is tasks["span"].generator
    resumed_settings = replace(settings, out_dir=str(tmp_path / "resumed"))
    resumed_records = train(resumed, tasks, resumed_settings, device=torch.device("cpu"), resume_from=checkpoint)
    interrupted = [json.loads(line) for line in
                   (tmp_path / "interrupted" / "train_log.jsonl").read_text().splitlines()]
    combined = interrupted + resumed_records
    assert [record["step"] for record in combined] == list(range(6))
    assert {record["task"] for record in full_records} == {"mlm", "span"}
    for full, actual in zip(full_records, combined):
        assert {key: value for key, value in full.items() if key != "seconds"} == {
            key: value for key, value in actual.items() if key != "seconds"
        }
    assert_state_equal(uninterrupted.state_dict(), resumed.state_dict())
    full = torch.load(tmp_path / "full" / "training_state" / "final.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "training_state" / "final.pt", weights_only=False)
    assert full["extra"]["settings"]["native_gqa"] is True
    assert_state_equal(full["optimizer"], resumed["optimizer"])
    assert_state_equal(full["extra"]["totals"], resumed["extra"]["totals"])


def test_resume_refuses_switching_native_execution_to_portable(interrupted_run, tmp_path):
    corpus, config, settings, checkpoint = interrupted_run(True)
    tasks, _ = base_stage(corpus, config, batch_size=2, seed=11)
    altered = replace(settings, out_dir=str(tmp_path / "changed"), native_gqa=False)
    with pytest.raises(ValueError, match="native_gqa"):
        train(NuriModel(config), tasks, altered, device=torch.device("cpu"), resume_from=checkpoint)


def test_legacy_missing_native_setting_means_false_and_refuses_true(interrupted_run, tmp_path):
    corpus, config, settings, checkpoint = interrupted_run(False)
    state = torch.load(checkpoint, weights_only=False)
    del state["extra"]["settings"]["native_gqa"]
    legacy = tmp_path / "legacy.pt"
    torch.save(state, legacy)
    tasks, _ = base_stage(corpus, config, batch_size=2, seed=11)
    enabled = replace(settings, out_dir=str(tmp_path / "legacy-enabled"), native_gqa=True)
    with pytest.raises(ValueError, match="native_gqa"):
        train(NuriModel(config), tasks, enabled, device=torch.device("cpu"), resume_from=legacy)
    tasks, _ = base_stage(corpus, config, batch_size=2, seed=11)
    default = replace(settings, out_dir=str(tmp_path / "legacy-default"))
    model = NuriModel(config)
    model.set_native_gqa(True)  # train() must reset any inherited runtime flag
    records = train(model, tasks, default, device=torch.device("cpu"), resume_from=legacy)
    assert [record["step"] for record in records] == [2, 3, 4, 5]
    assert all(not layer.native_gqa for layer in attention_layers(model))
