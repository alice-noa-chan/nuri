"""Opt-in recipes preserve defaults, control real batches and bind exact resumes."""

import copy
import json
import random
import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.data import TokenCorpus, build_mlm_batch, build_span_batch, write_corpus
from nuri.pipeline import base_stage, embedding_stage, instruction_stage
from nuri.recipe import (TrainingRecipe, load_recipe_config, recipe_loss_settings,
                         validate_recipe_resume)
from nuri.model import NuriModel
from scripts import train as cli


def recipe(stage="base", **fields):
    return TrainingRecipe.from_dict({"schema_version": 1, "stage": stage, **fields}, stage)


def save_recipe(tmp_path, data, name="recipe.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def corpus(tmp_path):
    path = tmp_path / "corpus"
    if not path.exists():
        write_corpus(path, [[3 + (row + position) % 100 for position in range(40 + row)] for row in range(12)])
    return TokenCorpus(path)


def same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("fields", [
    {"unknown": 1}, {"schema_version": True}, {"stage": "it"}, {"task_mix": {}},
    {"task_mix": {"mlm": 0}}, {"task_mix": {"mlm": -1}}, {"task_mix": {"pairs": 1}},
    {"task_mix": {"mlm": float("nan")}}, {"task_mix": {"mlm": 1e308, "span": 1e308}},
    {"corruption": {"mlm_rate": 0}}, {"corruption": {"mlm_rate": 1.01}},
    {"corruption": {"mlm_rate": True}}, {"corruption": {"min_span": 1.5}},
    {"corruption": {"canvas_full_mask_prob": 0.2}}, {"loss": {"length_weight": 0.2}},
    {"loss": None}, {"task_mix": None},
])
def test_unknown_stage_and_invalid_recipe_values_fail(fields):
    data = {"schema_version": 1, "stage": "base", **fields}
    with pytest.raises(ValueError):
        TrainingRecipe.from_dict(data, "base")


@pytest.mark.parametrize("loss", [
    {"temperature": 0}, {"temperature": float("inf")}, {"cross_mode_weight": -0.1},
    {"embed_dims": []}, {"embed_dims": [16, 16]}, {"embed_dims": [32, 16]},
    {"embed_dims": [True]}, {"embed_dims": list(range(1, 10))},
])
def test_invalid_embedding_loss_controls_fail(loss):
    with pytest.raises(ValueError):
        recipe("embed", loss=loss)


def test_canonical_recipe_is_immutable_roundtrippable_and_path_independent(tmp_path):
    data = {"schema_version": 1, "stage": "base", "task_mix": {"span": 0.4, "mlm": 0.6},
            "corruption": {"min_span": 8, "mlm_rate": 0.2}}
    declared = load_recipe_config(save_recipe(tmp_path, data), "base")
    reordered = {"corruption": {"mlm_rate": 0.2, "min_span": 8}, "task_mix": {"mlm": 0.6, "span": 0.4},
                 "stage": "base", "schema_version": 1}
    moved = load_recipe_config(save_recipe(tmp_path, reordered, "moved.json"), "base")
    assert declared.sha256 == moved.sha256
    assert list(declared.task_mix) == ["mlm", "span"]
    assert recipe().sha256 == TrainingRecipe.from_dict(recipe().to_dict(), "base").sha256
    with pytest.raises(TypeError):
        declared.corruption["mlm_rate"] = 1.0
    snapshot = declared.to_dict()
    snapshot["corruption"]["mlm_rate"] = 1.0
    assert declared.corruption["mlm_rate"] == 0.2
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"stage":"base","stage":"base"}')
    with pytest.raises(ValueError, match="duplicate recipe field"):
        load_recipe_config(duplicate, "base")


def test_no_config_keeps_legacy_batches_rng_and_task_options(tmp_path):
    config, source = tiny_config(), corpus(tmp_path)
    tasks, mix = base_stage(source, config, 3, seed=9)
    assert mix == {"mlm": 0.5, "span": 0.5}
    assert tasks["mlm"].options == {"batch_size": 3, "corpus_documents": len(source),
                                    "corpus_weights": None, "pack_mlm": False, "bucket": False}
    legacy_rng, legacy_generator = random.Random(9), torch.Generator().manual_seed(9)
    for name in ("mlm", "span", "mlm", "span"):
        if name == "mlm":
            documents = source.sample(3, legacy_rng, usable=lambda d: any(t not in (0, 1, 2) for t in d))
            expected = build_mlm_batch(documents, config, legacy_rng, legacy_generator)
        else:
            documents = source.sample(3, legacy_rng, usable=lambda d: len(d) >= 2)
            expected = build_span_batch(documents, config, legacy_rng, legacy_generator)
        same(tasks[name].next_batch(), expected)
    same(tasks["mlm"].rng.getstate(), legacy_rng.getstate())
    same(tasks["mlm"].generator.get_state(), legacy_generator.get_state())


@pytest.mark.parametrize("pack,bucket", [(False, False), (True, False), (False, True), (True, True)])
def test_explicit_default_recipe_keeps_packed_and_bucketed_rng(tmp_path, pack, bucket):
    config, source = tiny_config(), corpus(tmp_path)
    plain, default_mix = base_stage(source, config, 3, seed=8, pack_mlm=pack, bucket=bucket)
    declared = recipe(task_mix={"mlm": 0.5, "span": 0.5}, corruption={"mlm_rate": 0.3, "min_span": 16})
    configured, configured_mix = base_stage(source, config, 3, seed=8, pack_mlm=pack, bucket=bucket, recipe=declared)
    assert configured_mix == default_mix
    for name in ("mlm", "span", "mlm"):
        same(plain[name].next_batch(), configured[name].next_batch())
    same(plain["mlm"].rng.getstate(), configured["mlm"].rng.getstate())
    same(plain["mlm"].generator.get_state(), configured["mlm"].generator.get_state())
    assert "recipe" not in plain["mlm"].options
    assert configured["span"].options["recipe"] == declared.to_dict()


@pytest.mark.parametrize("reasoning,with_corpus", [(False, False), (True, False), (False, True), (True, True)])
def test_it_defaults_and_explicit_mix_never_redistribute_missing_tasks(tmp_path, reasoning, with_corpus):
    config = tiny_config()
    examples = [([3, 4], [5, 6]), ([7, 8], [9])]
    reason = [([10], [11])] if reasoning else None
    source = corpus(tmp_path) if with_corpus else None
    tasks, default = instruction_stage(examples, config, 2, source, reason)
    assert default == {"instruction": 0.7 + (0 if reasoning else 0.15) + (0 if with_corpus else 0.15),
                       **({"reasoning": 0.15} if reasoning else {}), **({"span": 0.15} if with_corpus else {})}
    selected = recipe("it", task_mix={name: index + 1 for index, name in enumerate(tasks)})
    configured, mix = instruction_stage(examples, config, 2, source, reason, recipe=selected)
    assert mix == dict(selected.task_mix)
    assert all(task.options["recipe_sha256"] == selected.sha256 for task in configured.values())
    if not reasoning:
        missing = recipe("it", task_mix={"instruction": 0.8, "reasoning": 0.2})
        with pytest.raises(ValueError, match="unavailable tasks"):
            instruction_stage(examples, config, 2, source, reason, recipe=missing)
        heldout, _ = instruction_stage(examples, config, 2, source, reason, recipe=missing, apply_recipe_mix=False)
        assert "reasoning" not in heldout


@pytest.mark.parametrize("pack", [False, True])
def test_recipe_mask_rate_controls_real_mlm_targets(tmp_path, pack):
    source, config = corpus(tmp_path), tiny_config()
    declared = recipe(task_mix={"mlm": 1}, corruption={"mlm_rate": 1})
    tasks, mix = base_stage(source, config, 3, recipe=declared, pack_mlm=pack)
    batch = tasks["mlm"].next_batch()
    assert mix == {"mlm": 1.0}
    assert torch.equal(batch["target_mask"], batch["input_valid"])
    assert "span" in tasks  # available assessments are not removed by sampling weights


def test_recipe_span_minimum_controls_real_targets(tmp_path):
    source = corpus(tmp_path)
    # Short tails remain legal; compare exact existing-builder geometry as well.
    config = tiny_config(max_response_len=40)
    plain, _ = base_stage(source, config, 10, seed=11)
    changed, _ = base_stage(source, config, 10, seed=11, recipe=recipe(corruption={"min_span": 32}))
    batch = changed["span"].next_batch()
    lengths = batch["canvas_valid"].sum(1)
    assert not torch.equal(lengths, plain["span"].next_batch()["canvas_valid"].sum(1))
    assert (lengths <= config.max_response_len).all()
    rng, generator = random.Random(11), torch.Generator().manual_seed(11)
    documents = source.sample(10, rng, usable=lambda d: len(d) >= 2)
    same(batch, build_span_batch(documents, config, rng, generator, min_span=32))


def cli_arguments(tmp_path, *, stage="base", output="out", extra=()):
    args = ["--stage", stage, "--out", str(tmp_path / output), "--steps", "3", "--batch-size", "2",
            "--no-progress", "--save-every", "1", "--log-every", "1", *extra]
    if stage == "base":
        return args + ["--preset", "tiny", "--corpus", str(tmp_path / "corpus")]
    return args + ["--init", str(tmp_path / "initial"), "--instructions", str(tmp_path / "instructions.jsonl"),
                   "--pairs", str(tmp_path / "pairs.jsonl"), "--reasoning-examples", "0"]


@pytest.mark.parametrize("fields,extra", [
    ({"corruption": {"mlm_rate": -1}}, ()), ({"corruption": {"min_span": 17}}, ()),
    ({"task_mix": {"not_a_task": 1}}, ()),
])
def test_invalid_cli_recipe_refused_before_model_allocation(tmp_path, monkeypatch, fields, extra):
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "base", **fields})
    monkeypatch.setattr(cli, "NuriModel", lambda *a, **k: pytest.fail("model allocated"))
    with pytest.raises(ValueError):
        cli.main(cli_arguments(tmp_path, extra=(*extra, "--recipe-config", str(path))))
    assert not (tmp_path / "out").exists()


def test_missing_explicit_it_inputs_and_embed_geometry_fail_before_loading(tmp_path, monkeypatch):
    def no_model(*args, **kwargs):
        pytest.fail("checkpoint/model allocated")
    monkeypatch.setattr(cli, "load_pretrained", no_model)
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "it", "task_mix": {"reasoning": 1}})
    with pytest.raises(ValueError, match="unavailable tasks"):
        cli.main(cli_arguments(tmp_path, stage="it", extra=("--recipe-config", str(path))))
    initial = tmp_path / "initial"
    initial.mkdir()
    tiny_config().save(initial / "config.json")
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "embed", "loss": {"embed_dims": [64]}})
    with pytest.raises(ValueError, match="embedding dimension"):
        cli.main(cli_arguments(tmp_path, stage="embed", extra=("--recipe-config", str(path))))
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "it", "corruption": {"min_span": 3}})
    with pytest.raises(ValueError, match="span corpus"):
        cli.main(cli_arguments(tmp_path, stage="it", extra=("--recipe-config", str(path))))


def test_recipe_env_precedence_and_cross_mode_conflict(tmp_path, monkeypatch):
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "base"})
    monkeypatch.setenv("NURI_RECIPE_CONFIG", str(path))
    assert cli.parse_args(cli_arguments(tmp_path)).recipe_config == str(path)
    assert cli.parse_args(cli_arguments(tmp_path, extra=("--recipe-config", "explicit.json"))).recipe_config == "explicit.json"
    declared = recipe("embed", loss={"embed_dims": [8, 16, 32], "temperature": 0.1, "cross_mode_weight": 0.2})
    assert recipe_loss_settings(declared)["cross_mode_weight"] == 0.2
    assert recipe_loss_settings(None) == {"cross_mode_weight": 0.0}
    with pytest.raises(ValueError, match="conflicts"):
        recipe_loss_settings(declared, 0.0, True)
    assert recipe_loss_settings(declared, 0.2, True)["embed_dims"] == (8, 16, 32)
    embed_tasks, _ = embedding_stage([([3], [4]), ([5], [6])], tiny_config(), 2, recipe=declared)
    assert embed_tasks["pairs"].options["recipe_sha256"] == declared.sha256


def test_default_cli_recipe_keeps_exact_weights_optimizer_rng(tmp_path):
    corpus(tmp_path)
    cli.main(cli_arguments(tmp_path, output="plain"))
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "base",
                                 "task_mix": {"mlm": 0.5, "span": 0.5},
                                 "corruption": {"mlm_rate": 0.3, "min_span": 16}})
    cli.main(cli_arguments(tmp_path, output="configured", extra=("--recipe-config", str(path))))
    same(load_file(str(tmp_path / "plain/final/model.safetensors")),
         load_file(str(tmp_path / "configured/final/model.safetensors")))
    left = torch.load(tmp_path / "plain/training_state/final.pt", weights_only=False)
    right = torch.load(tmp_path / "configured/training_state/final.pt", weights_only=False)
    same(left["optimizer"], right["optimizer"])
    same(left["extra"]["random"], right["extra"]["random"])
    same(left["extra"]["totals"], right["extra"]["totals"])


def test_recipe_cli_resume_is_exact_and_changed_or_legacy_bindings_fail_early(tmp_path, monkeypatch):
    corpus(tmp_path)
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "base", "task_mix": {"mlm": 0.7, "span": 0.3},
                                 "corruption": {"mlm_rate": 0.6, "min_span": 8}})
    cli.main(cli_arguments(tmp_path, output="original", extra=("--recipe-config", str(path))))
    state_path = tmp_path / "original/training_state/step-1.pt"
    cli.main(cli_arguments(tmp_path, output="resumed", extra=("--recipe-config", str(path), "--resume", str(state_path))))
    same(load_file(str(tmp_path / "original/final/model.safetensors")),
         load_file(str(tmp_path / "resumed/final/model.safetensors")))
    left = torch.load(tmp_path / "original/training_state/final.pt", weights_only=False)
    right = torch.load(tmp_path / "resumed/training_state/final.pt", weights_only=False)
    same(left["optimizer"], right["optimizer"])
    same(left["extra"]["random"], right["extra"]["random"])
    monkeypatch.setattr(cli, "NuriModel", lambda *a, **k: pytest.fail("model allocated before binding check"))
    changed = json.loads(path.read_text())
    changed["corruption"]["mlm_rate"] = 0.5
    altered = save_recipe(tmp_path, changed, "changed.json")
    with pytest.raises(ValueError, match="same recipe"):
        cli.main(cli_arguments(tmp_path, output="changed", extra=("--recipe-config", str(altered), "--resume", str(state_path))))
    for version in ("no_recipe", "no_options"):
        state = torch.load(state_path, weights_only=False)
        if version == "no_options":
            state["extra"].pop("task_options")
        else:
            for options in state["extra"]["task_options"].values():
                options.pop("recipe")
                options.pop("recipe_sha256")
        legacy = tmp_path / f"{version}.pt"
        torch.save(state, legacy)
        with pytest.raises(ValueError, match="recipe"):
            cli.main(cli_arguments(tmp_path, output=version, extra=("--recipe-config", str(path), "--resume", str(legacy))))


def test_recipe_binding_and_corruption_reach_every_language_holdout(tmp_path):
    corpus(tmp_path)
    args = cli.parse_args(cli_arguments(tmp_path))
    config = tiny_config()
    declared = recipe(task_mix={"mlm": 1}, corruption={"mlm_rate": 1, "min_span": 8})
    data = {"config_sha256": "1" * 64, "batch_size": 2}
    block = {"corpus": [str(tmp_path / "corpus")], "seed": 2, "depth": 2, "batches_per_task": 1,
             "language_groups": {language: {"corpus": [str(tmp_path / "corpus")]} for language in ("ko", "en", "ja")}}
    data.update(valid=copy.deepcopy(block), test=copy.deepcopy(block))
    validation, test = cli.build_holdouts(data, args, config, None, declared)
    for holdout in (validation, test):
        assert set(holdout.tasks) == {prefix + kind for prefix in ("", "ko/", "en/", "ja/") for kind in ("mlm", "span")}
        assert holdout.metadata["recipe_sha256"] == declared.sha256
        for name, task in holdout.tasks.items():
            assert task.options["recipe"] == declared.to_dict()
            if name.endswith("mlm"):
                batch = task.next_batch()
                assert torch.equal(batch["target_mask"], batch["input_valid"])


def initial_checkpoint(tmp_path, length_trained=False):
    words = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"]
    words += [f"word{i}" for i in range(120)]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    torch.manual_seed(4)
    save_pretrained(NuriModel(tiny_config(length_head_trained=length_trained)), tmp_path / "initial",
                    tokenizer_dir=tmp_path)
    examples = [{"messages": [{"role": "user", "content": f"word{row} word{row + 1}"}],
                 "response": f"word{row + 20} word{row + 21}"} for row in range(8)]
    pairs = [{"query": f"word{row} word{row + 1}", "positive": f"word{row + 20} word{row + 21}"}
             for row in range(8)]
    (tmp_path / "instructions.jsonl").write_text("\n".join(map(json.dumps, examples)), encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text("\n".join(map(json.dumps, pairs)), encoding="utf-8")


@pytest.mark.parametrize("stage,loss", [
    ("it", {"length_weight": 0.4}),
    ("embed", {"embed_dims": [8, 16, 32], "temperature": 0.2, "cross_mode_weight": 0.3}),
])
def test_cli_recipe_loss_controls_drive_real_components_and_resume(tmp_path, stage, loss):
    initial_checkpoint(tmp_path)
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": stage, "loss": loss})
    extra = ("--recipe-config", str(path))
    cli.main(cli_arguments(tmp_path, stage=stage, output="original", extra=extra))
    rows = [json.loads(line) for line in (tmp_path / "original/train_log.jsonl").read_text().splitlines()]
    settings = json.loads((tmp_path / "original/train_settings.json").read_text())
    for name, value in loss.items():
        assert settings[name] == value
    for row in rows:
        expected = row["denoise"] + 0.4 * row["length"] if stage == "it" else (
            (row["fast"] + row["reasoning"]) / 2 + 0.3 * row["cross_mode"])
        assert row["loss"] == pytest.approx(expected, rel=2e-6)
    state_path = tmp_path / "original/training_state/step-1.pt"
    cli.main(cli_arguments(tmp_path, stage=stage, output="resumed", extra=(*extra, "--resume", str(state_path))))
    same(load_file(str(tmp_path / "original/final/model.safetensors")),
         load_file(str(tmp_path / "resumed/final/model.safetensors")))


@pytest.mark.parametrize("previously_trained", [False, True])
def test_disabled_length_objective_does_not_advertise_new_training(tmp_path, previously_trained):
    initial_checkpoint(tmp_path, length_trained=previously_trained)
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "it", "loss": {"length_weight": 0}})
    cli.main(cli_arguments(tmp_path, stage="it", extra=("--recipe-config", str(path))))
    config = json.loads((tmp_path / "out/final/config.json").read_text())
    assert config["length_head_trained"] is previously_trained


def test_span_only_it_recipe_does_not_train_natural_length_head(tmp_path):
    initial_checkpoint(tmp_path)
    corpus(tmp_path)
    path = save_recipe(tmp_path, {"schema_version": 1, "stage": "it", "task_mix": {"span": 1}})
    cli.main(cli_arguments(tmp_path, stage="it", extra=("--recipe-config", str(path),
                                                       "--corpus", str(tmp_path / "corpus"))))
    config = json.loads((tmp_path / "out/final/config.json").read_text())
    assert not config["length_head_trained"]
    rows = [json.loads(line) for line in (tmp_path / "out/train_log.jsonl").read_text().splitlines()]
    assert all(row["task"] == "span" and "length" not in row for row in rows)


def test_resume_helper_rejects_partial_or_inconsistent_receipts():
    declared = recipe()
    good = {"recipe": declared.to_dict(), "recipe_sha256": declared.sha256}
    validate_recipe_resume(declared, {"task_options": {"mlm": good, "span": good}})
    for bad in ({}, {"recipe": declared.to_dict()}, {**good, "recipe_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="recipe"):
            validate_recipe_resume(declared, {"task_options": {"mlm": good, "span": bad}})
