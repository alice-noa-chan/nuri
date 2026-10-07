"""span_end_prob: spans that end at the document end so EOS is trained (Q-17)."""

import random

import pytest
import torch

from nuri import data
from nuri.config import tiny_config
from nuri.corruption import split_span
from nuri.pipeline import base_stage
from nuri.recipe import TrainingRecipe

EOS = 2
DOCUMENT = list(range(3, 63))  # 60 tokens


def draws(end_prob=None, count=40, seed=7):
    rng = random.Random(seed)
    kwargs = {} if end_prob is None else {"end_prob": end_prob}
    pairs = [split_span(DOCUMENT, 32, 16, EOS, 8, rng, **kwargs) for _ in range(count)]
    return pairs, rng.random()


def test_zero_end_prob_keeps_the_original_sampling_exactly():
    assert draws(0.0) == draws(None)


def test_full_end_prob_always_ends_at_the_document_end_with_eos():
    pairs, _ = draws(1.0)
    for source, target in pairs:
        content = target[:-1]
        assert target[-1] == EOS and EOS not in content
        assert 8 <= len(content) <= 15  # min_span .. M - 1, room for EOS
        assert DOCUMENT[-len(content) :] == content
        cut = len(DOCUMENT) - len(content)
        assert source == DOCUMENT[max(0, cut - 32) : cut]


def test_partial_end_prob_mixes_both_kinds():
    pairs, _ = draws(0.5, count=200)
    with_eos = sum(target[-1] == EOS for _, target in pairs)
    assert 60 < with_eos < 160


def test_short_documents_still_leave_an_input_token():
    for length in (2, 3, 5):
        document = list(range(3, 3 + length))
        source, target = split_span(
            document, 32, 16, EOS, 8, random.Random(0), end_prob=1.0
        )
        assert len(source) >= 1 and target[-1] == EOS
        assert source + target[:-1] == document


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.5", None])
def test_invalid_end_prob_is_rejected(bad):
    with pytest.raises(ValueError):
        split_span(DOCUMENT, 32, 16, EOS, 8, random.Random(0), end_prob=bad)


def test_span_batch_passes_the_option():
    config = tiny_config()
    documents = [list(range(3, 3 + n)) for n in range(20, 40)]
    batch = data.build_span_batch(
        documents,
        config,
        random.Random(0),
        torch.Generator().manual_seed(0),
        span_end_prob=1.0,
    )
    lengths = batch["canvas_valid"].sum(dim=1)
    last = batch["targets"].gather(1, (lengths - 1)[:, None]).squeeze(1)
    assert (last == config.eos_id).all()


def recipe(stage, value):
    return TrainingRecipe.from_dict(
        {"schema_version": 1, "stage": stage, "corruption": {"span_end_prob": value}},
        stage,
    )


def test_recipe_accepts_and_validates_span_end_prob():
    for stage in ("base", "it"):
        assert recipe(stage, 0.25).corruption["span_end_prob"] == 0.25
    with pytest.raises(ValueError):
        recipe("base", 1.5)
    with pytest.raises(ValueError):
        recipe("embed", 0.25)


def test_base_stage_recipe_reaches_span_batches(tmp_path):
    folder = tmp_path / "corpus"
    data.write_corpus(
        folder, [[3 + (row + i) % 120 for i in range(38 + row)] for row in range(24)]
    )
    config = tiny_config()
    tasks, _ = base_stage(
        data.open_corpus(str(folder)), config, 4, recipe=recipe("base", 1.0)
    )
    batch = tasks["span"].next_batch()
    lengths = batch["canvas_valid"].sum(dim=1)
    last = batch["targets"].gather(1, (lengths - 1)[:, None]).squeeze(1)
    assert (last == config.eos_id).all()
