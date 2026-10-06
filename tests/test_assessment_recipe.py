"""Independent corruption comparators preserve training and bind exact assessment resumes."""

import copy
import json
import random

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.recipe import TrainingRecipe
from nuri.trainer import TrainSettings, _resume, _save_state, evaluate_holdout
from scripts import train as cli


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def write_recipe(tmp_path, declared_stage="base", name="assessment.json", **fields):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "stage": declared_stage, **fields}), encoding="utf-8")
    return path


def fixture_inputs(tmp_path):
    words = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"]
    words += [f"word{i}" for i in range(120)]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(128))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    torch.manual_seed(4)
    save_pretrained(NuriModel(tiny_config()), tmp_path / "initial", tokenizer_dir=tmp_path)
    config = {"schema_version": 1, "valid_every": 0, "batch_size": 3}
    for shift, split in enumerate(("train", "valid", "test")):
        folder = tmp_path / split
        write_corpus(folder, [[8 + (shift * 11 + row + position) % 100 for position in range(50 + row)]
                             for row in range(12)])
        examples = [{"messages": [{"role": "user", "content": f"word{row} word{shift}"}],
                     "response": " ".join(f"word{i}" for i in range(20 + row, 30 + row))} for row in range(12)]
        (folder / "instructions.jsonl").write_text("\n".join(map(json.dumps, examples)), encoding="utf-8")
        if split != "train":
            resources = {"corpus": [split], "instructions": f"{split}/instructions.jsonl",
                         "reasoning": f"{split}/instructions.jsonl"}
            config[split] = {**resources, "batches_per_task": 1, "seed": 3, "depth": 2,
                             "language_groups": {name: copy.deepcopy(resources) for name in ("ko", "en", "ja")}}
    (tmp_path / "holdout.json").write_text(json.dumps(config), encoding="utf-8")
    return tokenizer


def arguments(tmp_path, *, stage="base", extra=()):
    values = ["--stage", stage, "--out", str(tmp_path / "out"), "--steps", "4", "--batch-size", "3",
              "--holdout-config", str(tmp_path / "holdout.json"), "--no-progress", *extra]
    if stage == "base":
        return values + ["--preset", "tiny", "--corpus", str(tmp_path / "train")]
    return values + ["--init", str(tmp_path / "initial"), "--instructions", str(tmp_path / "train/instructions.jsonl"),
                     "--corpus", str(tmp_path / "train"), "--reasoning-examples", "0"]


def assert_batch_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]) if torch.is_tensor(left[key]) else left[key] == right[key]


def test_assessment_flag_requires_holdout_and_cli_overrides_environment(monkeypatch):
    common = ["--stage", "base", "--out", "unused", "--steps", "4"]
    monkeypatch.delenv("NURI_HOLDOUT_CONFIG", raising=False)
    with pytest.raises(SystemExit) as error:
        cli.parse_args(common + ["--assessment-recipe-config", "unused.json"])
    assert error.value.code == 2
    monkeypatch.setenv("NURI_HOLDOUT_CONFIG", "holdout.json")
    monkeypatch.setenv("NURI_ASSESSMENT_RECIPE_CONFIG", "environment.json")
    assert cli.parse_args(common).assessment_recipe_config == "environment.json"
    assert cli.parse_args(common + ["--assessment-recipe-config", "explicit.json"]).assessment_recipe_config == "explicit.json"


@pytest.mark.parametrize("stage,fields", [
    ("base", {"task_mix": {"mlm": 1}}), ("it", {"loss": {"length_weight": 0}}),
    ("base", {"corruption": {"full_mask_prob": float("nan")}}),
    ("base", {"corruption": {"min_span": 17}}), ("base", {"unknown": 1}),
    ("base", {"stage": "it"}),
])
def test_invalid_assessment_fails_before_model_allocation(tmp_path, monkeypatch, stage, fields):
    fixture_inputs(tmp_path)
    path = write_recipe(tmp_path, stage, **fields)
    monkeypatch.setattr(cli, "NuriModel", lambda *a, **k: pytest.fail("model allocated"))
    monkeypatch.setattr(cli, "load_pretrained", lambda *a, **k: pytest.fail("checkpoint allocated"))
    with pytest.raises(ValueError):
        cli.main(arguments(tmp_path, stage=stage, extra=("--assessment-recipe-config", str(path))))
    assert not (tmp_path / "out").exists()


def test_it_minimum_requires_span_in_every_assessment_group(tmp_path):
    fixture_inputs(tmp_path)
    path = write_recipe(tmp_path, "it", corruption={"min_span": 8})
    args = cli.parse_args(arguments(tmp_path, stage="it", extra=("--assessment-recipe-config", str(path))))
    data = cli.load_holdout_config(args)
    del data["valid"]["language_groups"]["ja"]["corpus"]
    with pytest.raises(ValueError, match="every IT holdout group"):
        cli.prepare_assessment_recipe(args, tiny_config(), data)


@pytest.mark.parametrize("declared", [None, {"full_mask_prob": 1}])
def test_absent_override_preserves_inherited_holdout_tasks_rng_and_metadata(tmp_path, declared):
    tokenizer = fixture_inputs(tmp_path)
    args = cli.parse_args(arguments(tmp_path))
    recipe = None if declared is None else TrainingRecipe.from_dict(
        {"schema_version": 1, "stage": "base", "corruption": declared}, "base")
    data = cli.load_holdout_config(args)
    left = cli.build_holdouts(data, args, tiny_config(), tokenizer, recipe)
    right = cli.build_holdouts(data, args, tiny_config(), tokenizer, recipe, assessment_recipe=None)
    for a, b in zip(left, right):
        assert a.metadata == b.metadata and a.tasks.keys() == b.tasks.keys()
        assert "assessment_recipe_sha256" not in a.metadata
        for name in a.tasks:
            assert a.tasks[name].options == b.tasks[name].options
            assert_batch_equal(a.tasks[name].next_batch(), b.tasks[name].next_batch())
            assert a.tasks[name].rng.getstate() == b.tasks[name].rng.getstate()
            assert torch.equal(a.tasks[name].generator.get_state(), b.tasks[name].generator.get_state())


@pytest.mark.parametrize("stage", ["base", "it"])
def test_cli_independent_policy_keeps_training_mix_losses_and_all_language_objectives(tmp_path, monkeypatch, stage):
    fixture_inputs(tmp_path)
    training_fields = {"corruption": {"full_mask_prob": 1},
                       "task_mix": {"mlm": 1} if stage == "base" else {"instruction": 1}}
    if stage == "it":
        training_fields["loss"] = {"length_weight": 0.6}
    training_path = write_recipe(tmp_path, stage, "training.json", **training_fields)
    fixed_path = write_recipe(tmp_path, stage, corruption={"full_mask_prob": 0}, loss={})
    captured = {}

    def capture(model, tasks, settings, **kwargs):
        captured.update(model=model, tasks=tasks, settings=settings, **kwargs)
        return []

    monkeypatch.setattr(cli, "train", capture)
    cli.main(arguments(tmp_path, stage=stage, extra=("--recipe-config", str(training_path),
                                                    "--assessment-recipe-config", str(fixed_path))))
    settings = captured["settings"]
    assert settings.mix == training_fields["task_mix"]
    assert settings.length_weight == (0.6 if stage == "it" else 0.1)
    training_recipe = TrainingRecipe.from_dict({"schema_version": 1, "stage": stage, **training_fields}, stage)
    fixed_recipe = TrainingRecipe.from_dict({"schema_version": 1, "stage": stage, "loss": {},
                                            "corruption": {"full_mask_prob": 0}}, stage)
    for task in captured["tasks"].values():
        assert task.options["recipe_sha256"] == training_recipe.sha256
        assert task.options["assessment_recipe_sha256"] == fixed_recipe.sha256
        if task.kind != "mlm":
            batch = task.next_batch()
            assert torch.equal(batch["loss_mask"], batch["canvas_valid"])
    expected = ("mlm", "span") if stage == "base" else ("instruction", "reasoning", "span")
    for split in ("validation", "test"):
        holdout = captured[split]
        assert set(holdout.tasks) == {prefix + name for prefix in ("", "ko/", "en/", "ja/") for name in expected}
        assert holdout.metadata["recipe_sha256"] == fixed_recipe.sha256
        assert holdout.metadata["assessment_recipe_sha256"] == fixed_recipe.sha256
        assert all(task.options["recipe_sha256"] == fixed_recipe.sha256 for task in holdout.tasks.values())
    report = evaluate_holdout(captured["model"], captured["validation"], settings,
                              torch.device("cpu"), "valid", completed_steps=0)
    assert report["metadata"]["recipe_sha256"] == fixed_recipe.sha256
    assert report["objective_settings"]["length_weight"] == settings.length_weight
    assert all(parameter.grad is None for parameter in captured["model"].parameters())


def test_fixed_assessment_can_be_used_without_training_recipe(tmp_path, monkeypatch):
    fixture_inputs(tmp_path)
    path = write_recipe(tmp_path, corruption={"full_mask_prob": 0.1, "mlm_rate": 0.3, "min_span": 16})
    captured = {}

    def capture(model, tasks, settings, **kwargs):
        captured.update(tasks=tasks, settings=settings, **kwargs)
        return []

    monkeypatch.setattr(cli, "train", capture)
    cli.main(arguments(tmp_path, extra=("--assessment-recipe-config", str(path))))
    assert captured["settings"].mix == {"mlm": 0.5, "span": 0.5}
    assert all("recipe" not in task.options for task in captured["tasks"].values())
    assert all("assessment_recipe" in task.options for task in captured["tasks"].values())
    plain, _ = base_stage(TokenCorpus(tmp_path / "train"), tiny_config(), 3)
    for name in plain:
        assert_batch_equal(plain[name].next_batch(), captured["tasks"][name].next_batch())


def test_explicit_assessment_resume_requires_exact_saved_binding(tmp_path):
    fixture_inputs(tmp_path)
    path = write_recipe(tmp_path, corruption={"full_mask_prob": 0.1})
    declared = TrainingRecipe.from_dict({"schema_version": 1, "stage": "base",
                                        "corruption": {"full_mask_prob": 0.1}}, "base")
    tasks, _ = base_stage(TokenCorpus(tmp_path / "train"), tiny_config(), 3)
    cli.bind_assessment_recipe(tasks, declared)
    model = NuriModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    settings = TrainSettings(out_dir=str(tmp_path), total_steps=4, warmup_steps=1, progress=False)
    checkpoint = _save_state(model, optimizer, settings, 0, {"tokens": 1, "targets": 1},
                             tmp_path, random.Random(0), tasks, "saved")
    args = cli.parse_args(arguments(tmp_path, extra=("--assessment-recipe-config", str(path), "--resume", str(checkpoint))))
    assert cli.prepare_assessment_recipe(args).sha256 == declared.sha256
    moved = write_recipe(tmp_path, name="moved.json", corruption={"full_mask_prob": 0.10})
    args.assessment_recipe_config = str(moved)
    assert cli.prepare_assessment_recipe(args).sha256 == declared.sha256
    changed = write_recipe(tmp_path, name="changed.json", corruption={"full_mask_prob": 0.3})
    args.assessment_recipe_config = str(changed)
    with pytest.raises(ValueError, match="same saved assessment recipe"):
        cli.prepare_assessment_recipe(args)
    plain, _ = base_stage(TokenCorpus(tmp_path / "train"), tiny_config(), 3)
    with pytest.raises(ValueError, match="same batch construction"):
        _resume(checkpoint, model, optimizer, random.Random(0), plain, settings)

    saved = torch.load(checkpoint, weights_only=False)
    saved["extra"]["task_options"]["span"].pop("assessment_recipe_sha256")
    damaged = tmp_path / "damaged.pt"
    torch.save(saved, damaged)
    args.resume, args.assessment_recipe_config = str(damaged), str(path)
    with pytest.raises(ValueError, match="same saved assessment recipe"):
        cli.prepare_assessment_recipe(args)


def test_assessment_cannot_be_silently_added_to_legacy_resume(tmp_path, monkeypatch):
    fixture_inputs(tmp_path)
    path = write_recipe(tmp_path, corruption={"full_mask_prob": 0.1})
    state = tmp_path / "legacy.pt"
    torch.save({"config": tiny_config().to_dict(), "extra": {"task_options": {"span": {}}}}, state)
    monkeypatch.setattr(cli, "NuriModel", lambda *a, **k: pytest.fail("model allocated"))
    with pytest.raises(ValueError, match="same saved assessment recipe"):
        cli.main(arguments(tmp_path, extra=("--assessment-recipe-config", str(path), "--resume", str(state))))
    assert not (tmp_path / "out").exists()
