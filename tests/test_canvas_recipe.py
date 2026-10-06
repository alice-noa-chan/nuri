"""Canvas recipe controls use existing corruption and preserve exact legacy sampling."""

import copy
import random

import pytest
import torch

from nuri import data
from nuri.config import tiny_config
from nuri.model import NuriModel
from nuri.pipeline import base_stage, instruction_stage
from nuri.recipe import TrainingRecipe, validate_recipe_resume
from nuri.trainer import (HoldoutSet, TrainSettings, _holdout_policy_sha256,
                          _resume, _save_state)
from scripts import train as train_cli


def recipe(stage, probability=None):
    corruption = {} if probability is None else {"full_mask_prob": probability}
    return TrainingRecipe.from_dict({"schema_version": 1, "stage": stage,
                                     "corruption": corruption}, stage)


def inputs(tmp_path):
    folder = tmp_path / "corpus"
    data.write_corpus(folder, [[3 + (row + position) % 120 for position in range(38 + row)]
                               for row in range(24)])
    examples = [([3 + row % 20] * (row % 34 + 1), [30 + row % 20] * (row % 15 + 1))
                for row in range(24)]
    reasoning = [([70 + row % 20], [90 + row % 20] * (row % 15 + 1)) for row in range(24)]
    return data.TokenCorpus(folder), examples, reasoning


def stage_tasks(stage, source, examples, reasoning, *, bucket=False, pack=False, declared=None):
    config = tiny_config()
    if stage == "base":
        return base_stage(source, config, 3, seed=17, pack_mlm=pack, bucket=bucket, recipe=declared)[0]
    return instruction_stage(examples, config, 3, source, reasoning, seed=17,
                             bucket=bucket, recipe=declared)[0]


def assert_batch_equal(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value, right[key]), key
        else:
            assert value == right[key], key


def assert_sources_equal(left, right):
    for name in left:
        assert left[name].rng.getstate() == right[name].rng.getstate()
        assert torch.equal(left[name].generator.get_state(), right[name].generator.get_state())


@pytest.mark.parametrize("stage", ["base", "it"])
@pytest.mark.parametrize("probability", [-0.01, 1.01, float("inf"), float("nan"), True, "0.5", None])
def test_invalid_probability_rejected(stage, probability):
    with pytest.raises(ValueError, match="full_mask_prob"):
        TrainingRecipe.from_dict({"schema_version": 1, "stage": stage,
                                  "corruption": {"full_mask_prob": probability}}, stage)


def test_probability_stage_and_canonical_binding():
    with pytest.raises(ValueError, match="unsupported fields"):
        recipe("embed", 0.5)
    for stage in ("base", "it"):
        assert recipe(stage, 0).sha256 == recipe(stage, 0.0).sha256
        assert recipe(stage, 1).sha256 == recipe(stage, 1.0).sha256
        assert recipe(stage, 0.1).sha256 != recipe(stage).sha256
        declared = recipe(stage, 0.5)
        assert TrainingRecipe.from_dict(declared.to_dict(), stage).sha256 == declared.sha256
        with pytest.raises(TypeError):
            declared.corruption["full_mask_prob"] = 1.0


def test_invalid_probability_fails_before_model_allocation(tmp_path, monkeypatch):
    path = tmp_path / "recipe.json"
    path.write_text('{"schema_version":1,"stage":"base","corruption":{"full_mask_prob":2}}')
    monkeypatch.setattr(train_cli, "NuriModel", lambda *a, **k: pytest.fail("model allocated"))
    with pytest.raises(ValueError, match="full_mask_prob"):
        train_cli.main(["--stage", "base", "--preset", "tiny", "--corpus", str(tmp_path / "unused"),
                        "--out", str(tmp_path / "out"), "--steps", "4", "--recipe-config", str(path)])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("stage,bucket,pack", [
    ("base", False, False), ("base", True, False),
    ("base", False, True), ("base", True, True), ("it", False, False), ("it", True, False),
])
def test_absent_and_explicit_default_keep_legacy_batches_and_rng(tmp_path, monkeypatch, stage, bucket, pack):
    source, examples, reasoning = inputs(tmp_path)
    corrupt_canvas = data.corrupt_canvas

    def original_canvas_call(targets, valid, mask_id, *, generator, **unused):
        # The original builder left the probability argument absent.
        return corrupt_canvas(targets, valid, mask_id, generator=generator)

    names = ("mlm", "span", "mlm", "span") if stage == "base" else (
        "instruction", "reasoning", "span", "instruction", "span")
    with monkeypatch.context() as context:
        context.setattr(data, "corrupt_canvas", original_canvas_call)
        legacy = stage_tasks(stage, source, examples, reasoning, bucket=bucket, pack=pack)
        expected = [legacy[name].next_batch() for name in names]
    for declared in (None, recipe(stage), recipe(stage, 0.1)):
        current = stage_tasks(stage, source, examples, reasoning, bucket=bucket, pack=pack, declared=declared)
        for name, batch in zip(names, expected):
            assert_batch_equal(current[name].next_batch(), batch)
        assert_sources_equal(current, legacy)
        for name, task in current.items():
            if declared is None:
                assert task.options == legacy[name].options
            else:
                assert task.options["recipe_sha256"] == declared.sha256


@pytest.mark.parametrize("stage", ["base", "it"])
@pytest.mark.parametrize("bucket", [False, True])
def test_full_mask_recipe_reaches_every_canvas_task(tmp_path, stage, bucket):
    source, examples, reasoning = inputs(tmp_path)
    tasks = stage_tasks(stage, source, examples, reasoning, bucket=bucket, declared=recipe(stage, 1))
    default = stage_tasks(stage, source, examples, reasoning, bucket=bucket)
    for name, task in tasks.items():
        assert task.options["recipe"]["corruption"]["full_mask_prob"] == 1.0
        batch = task.next_batch()
        if name == "mlm":
            assert_batch_equal(batch, default[name].next_batch())
            continue
        valid = batch["canvas_valid"]
        assert torch.equal(batch["loss_mask"], valid)
        assert (batch["canvas"][valid] == tiny_config().mask_id).all()
        assert torch.equal(batch["canvas"][~valid], batch["targets"][~valid])
        assert (batch["targets"][~valid] == tiny_config().pad_id).all()
        if name in ("instruction", "reasoning"):
            assert (batch["targets"].gather(1, (batch["lengths"] - 1).unsqueeze(1)) == tiny_config().eos_id).all()


@pytest.mark.parametrize("bucket", [False, True])
def test_zero_probability_uses_existing_partial_branch(bucket):
    config = tiny_config()
    examples = [([3] * (i + 1), [10 + i] * 15) for i in range(5)]
    limits = (5 * config.max_input_len, 5 * config.max_response_len, 5) if bucket else None
    batch = data.build_instruction_batch(examples, config, torch.Generator().manual_seed(3),
                                         bucket=limits, rng=random.Random(3), full_mask_prob=0)
    counts = batch["loss_mask"].sum(1)
    assert (counts >= 1).all() and (counts < 16).all()
    assert (batch["canvas"][~batch["loss_mask"]] == batch["targets"][~batch["loss_mask"]]).all()
    short = data.build_instruction_batch([([3], [])], config, torch.Generator().manual_seed(3), full_mask_prob=0)
    assert short["targets"].tolist() == [[config.eos_id]]
    assert short["loss_mask"].all()  # At least one target even when the dedicated branch is disabled.


@pytest.mark.parametrize("stage", ["base", "it"])
def test_full_state_resume_restores_recipe_batches_and_rejects_changes(tmp_path, stage):
    source, examples, reasoning = inputs(tmp_path)
    declared = recipe(stage, 0.6)
    tasks = stage_tasks(stage, source, examples, reasoning, bucket=True, declared=declared)
    for task in tasks.values():
        task.next_batch()
    model = NuriModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    settings = TrainSettings(out_dir=str(tmp_path), total_steps=8, warmup_steps=1, progress=False)
    rng, totals = random.Random(9), {"tokens": 100, "targets": 30}
    checkpoint = _save_state(model, optimizer, settings, 2, totals, tmp_path, rng, tasks, "resume")
    expected = {name: task.next_batch() for name, task in tasks.items()}
    restored = stage_tasks(stage, source, examples, reasoning, bucket=True, declared=declared)
    step, restored_totals = _resume(checkpoint, model, optimizer, random.Random(0), restored, settings)
    assert step == 3 and restored_totals == totals
    for name in restored:
        assert_batch_equal(restored[name].next_batch(), expected[name])
    assert_sources_equal(tasks, restored)

    extra = torch.load(checkpoint, weights_only=False)["extra"]
    validate_recipe_resume(declared, extra)
    for changed in (recipe(stage, 0.1), recipe(stage)):
        with pytest.raises(ValueError, match="same recipe"):
            validate_recipe_resume(changed, extra)
        changed_tasks = stage_tasks(stage, source, examples, reasoning, bucket=True, declared=changed)
        with pytest.raises(ValueError, match="same batch construction"):
            _resume(checkpoint, model, optimizer, random.Random(0), changed_tasks, settings)


def test_independent_holdout_recipe_keeps_its_own_corruption_and_policy(tmp_path):
    source, examples, reasoning = inputs(tmp_path)
    training = stage_tasks("base", source, examples, reasoning, declared=recipe("base", 1))
    fixed = stage_tasks("base", source, examples, reasoning, declared=recipe("base", 0))
    holdout = HoldoutSet(fixed, batches_per_task=1)
    original_options = copy.deepcopy({name: task.options for name, task in fixed.items()})
    initial_policy = _holdout_policy_sha256(holdout)
    for _ in range(3):
        batch = training["span"].next_batch()
        assert torch.equal(batch["loss_mask"], batch["canvas_valid"])
    assert {name: task.options for name, task in fixed.items()} == original_options
    assert _holdout_policy_sha256(holdout) == initial_policy
    changed = HoldoutSet(stage_tasks("base", source, examples, reasoning, declared=recipe("base", 1)),
                         batches_per_task=1)
    assert _holdout_policy_sha256(changed) != initial_policy
    batch = fixed["span"].next_batch()
    assert not torch.equal(batch["loss_mask"], batch["canvas_valid"])
