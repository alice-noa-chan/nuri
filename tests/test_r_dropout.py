"""r_dropout_prob: hide R from the readout for some span samples during training."""

import random

import pytest
import torch

from nuri import data
from nuri.config import tiny_config
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.recipe import TrainingRecipe
from nuri.trainer import canvas_loss

DOCUMENTS = [list(range(3, 3 + n)) for n in range(20, 44)]


def span_batch(seed=0, **options):
    return data.build_span_batch(
        DOCUMENTS,
        tiny_config(),
        random.Random(seed),
        torch.Generator().manual_seed(seed),
        **options,
    )


def make_model():
    torch.manual_seed(0)
    return NuriModel(tiny_config())


def test_zero_probability_keeps_batches_identical():
    plain, explicit = span_batch(), span_batch(r_dropout_prob=0.0)
    assert set(plain) == set(explicit) and "hide_r" not in plain
    for key in plain:
        assert torch.equal(plain[key], explicit[key])


def test_hide_flags_follow_the_probability_and_leave_the_canvas_alone():
    plain = span_batch()
    hidden = span_batch(r_dropout_prob=1.0)
    assert hidden["hide_r"].all()
    for key in plain:
        assert torch.equal(plain[key], hidden[key])  # drawn after the corruption
    mixed = data.build_span_batch(
        DOCUMENTS * 20, tiny_config(), random.Random(0), torch.Generator().manual_seed(0),
        r_dropout_prob=0.5,
    )  # fmt: skip
    assert 0.3 < mixed["hide_r"].float().mean().item() < 0.7
    with pytest.raises(ValueError):
        span_batch(r_dropout_prob=1.5)


def test_training_loss_hides_r_and_evaluation_does_not():
    model = make_model()
    batch = span_batch(r_dropout_prob=1.0)
    model.train()
    hidden_loss = canvas_loss(model, batch, depth=1)["loss"]
    model.eval()
    eval_loss = canvas_loss(model, batch, depth=1)["loss"]
    plain = {k: v for k, v in batch.items() if k != "hide_r"}
    assert torch.allclose(eval_loss, canvas_loss(model, plain, depth=1)["loss"])
    assert not torch.allclose(hidden_loss, eval_loss)

    # Hidden R means the readout loss sends no gradient into the core.
    model.train()
    model.zero_grad()
    canvas_loss(model, batch, depth=1)["loss"].backward()
    core_grads = [p.grad for p in model.core.parameters()]
    assert all(g is None or not g.any() for g in core_grads)
    model.zero_grad()
    canvas_loss(model, plain, depth=1)["loss"].backward()
    assert any(p.grad is not None and p.grad.any() for p in model.core.parameters())


def test_recipe_allows_it_only_for_base_span_training():
    recipe = TrainingRecipe.from_dict(
        {"schema_version": 1, "stage": "base", "corruption": {"r_dropout_prob": 0.3}},
        "base",
    )
    assert recipe.corruption["r_dropout_prob"] == 0.3
    with pytest.raises(ValueError):
        TrainingRecipe.from_dict(
            {"schema_version": 1, "stage": "it", "corruption": {"r_dropout_prob": 0.3}},
            "it",
        )
    with pytest.raises(ValueError):
        TrainingRecipe.from_dict(
            {"schema_version": 1, "stage": "base", "corruption": {"r_dropout_prob": 2}},
            "base",
        )


def test_base_stage_recipe_reaches_span_batches(tmp_path):
    folder = tmp_path / "corpus"
    data.write_corpus(folder, DOCUMENTS)
    recipe = TrainingRecipe.from_dict(
        {"schema_version": 1, "stage": "base", "corruption": {"r_dropout_prob": 1.0}},
        "base",
    )
    tasks, _ = base_stage(
        data.open_corpus(str(folder)), tiny_config(), 4, recipe=recipe
    )
    assert tasks["span"].next_batch()["hide_r"].all()
    assert "hide_r" not in tasks["mlm"].next_batch()
