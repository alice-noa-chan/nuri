"""Check source attribution against the actual builders, using synthetic TRAIN."""

import json
import random

import pytest
import torch

from nuri.config import NuriConfig
from nuri.data import (
    MixedCorpus,
    TokenCorpus,
    build_mlm_batch,
    build_packed_mlm_batch,
    build_span_batch,
    write_corpus,
)
from nuri.pipeline import base_stage
from nuri.recipe import TrainingRecipe
from tools.audit_batch_policy import main, mixed_mlm, mixed_span, validate


def geometry():
    return NuriConfig(
        vocab_size=64,
        hidden_size=16,
        num_heads=2,
        ffn_size=32,
        input_layers=1,
        readout_layers=1,
        latent_slots=2,
        max_input_len=32,
        max_response_len=16,
        embed_dim=8,
        train_depths=(1,),
    )


@pytest.mark.parametrize("packed", [False, True])
def test_single_source_counts_match_direct_builder(tmp_path, packed):
    config = geometry()
    write_corpus(tmp_path / "train", [[8] * 5, [9] * 70, [10] * 19])
    corpus = TokenCorpus(tmp_path / "train")
    spec = {"batches": 3, "batch_size": 2, "seed": 17, "pack_mlm": packed}
    result = mixed_mlm([corpus], ["en"], [1.0], config, spec)
    rng, generator = random.Random(17), torch.Generator().manual_seed(17)
    positions = targets = 0
    for _ in range(3):

        def draw():
            rng.choices(range(1), [1.0])
            return corpus.random_document(rng)

        batch = (
            build_packed_mlm_batch(draw, 2, config, rng, generator)
            if packed
            else build_mlm_batch([draw(), draw()], config, rng, generator)
        )
        positions += int(batch["input_valid"].sum())
        targets += int(batch["target_mask"].sum())
    assert result["by_language"]["en"]["input_positions"] == positions
    assert result["by_language"]["en"]["supervised_targets"] == targets
    assert result["total_rows"] == 6


def test_mixed_packed_positions_remain_language_separated(tmp_path):
    corpora = []
    for language, token in (("ko", 8), ("en", 9)):
        write_corpus(tmp_path / language, [[token] * 7])
        corpora.append(TokenCorpus(tmp_path / language))
    report = mixed_mlm(
        corpora, ["ko", "en"], [1, 1], geometry(), {"seed": 3, "batches": 8}
    )
    groups = report["by_language"]
    assert all(
        group["input_positions"] > 0 and group["supervised_targets"] > 0
        for group in groups.values()
    )
    assert sum(group["input_positions"] for group in groups.values()) == 16 * 21
    assert all(group["rows"] is None for group in groups.values())


def test_cli_binds_inputs_and_preserves_existing_receipt(tmp_path):
    tmp_path = tmp_path / ".private"
    tmp_path.mkdir()
    config = geometry()
    config.save(tmp_path / "model.json")
    write_corpus(tmp_path / "train", [[8] * 12, [9] * 40])
    spec = {
        "schema_version": 1,
        "model_config": "model.json",
        "batches": 2,
        "corpora": [{"language": "ko", "path": "train", "weight": 1, "split": "train"}],
    }
    path, output = tmp_path / "audit.json", tmp_path / "receipt.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    main([str(path), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert (
        report["optimizer_updates"] == 0
        and report["gpu_used"] is False
        and report["test_used"] is False
    )
    assert len(report["input_sha256"]) == 4
    assert report["span_isolated_by_language"]["ko"]["supervised_targets"] > 0
    assert report["span_mixed"]["total_rows"] > 0
    prior = output.read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        main([str(path), "--output", str(output)])
    assert output.read_bytes() == prior


def test_cli_refuses_public_configuration_or_receipt(tmp_path):
    with pytest.raises(ValueError, match=".private"):
        main([str(tmp_path / "audit.json"), "--output", str(tmp_path / "receipt.json")])


@pytest.mark.parametrize(
    "change",
    [
        {"batches": True},
        {"batches": 513},
        {"batch_size": 0},
        {"mlm_rate": float("nan")},
        {"min_span": 0},
        {"unknown": 1},
    ],
)
def test_invalid_work_limits_rejected(change):
    spec = {
        "schema_version": 1,
        "model_config": "config.json",
        "corpora": [{"language": "en", "path": "train", "weight": 1, "split": "train"}],
        **change,
    }
    with pytest.raises(ValueError):
        validate(spec)


def test_nontraining_resources_rejected():
    spec = {
        "schema_version": 1,
        "model_config": "config.json",
        "corpora": [
            {"language": "en", "path": "heldout", "weight": 1, "split": "test"}
        ],
    }
    with pytest.raises(ValueError, match="split=train"):
        validate(spec)


@pytest.mark.parametrize("bucketed", [False, True])
def test_mixed_span_matches_production_rng_and_selected_token_counts(
    tmp_path, bucketed
):
    config = geometry()
    corpora = []
    for language, token, lengths in (("ko", 8, (3, 5, 13)), ("en", 9, (32, 48, 96))):
        write_corpus(tmp_path / language, [[token] * length for length in lengths])
        corpora.append(TokenCorpus(tmp_path / language))
    spec = {
        "seed": 7,
        "batches": 10,
        "batch_size": 2,
        "bucket_span": bucketed,
        "full_mask_prob": 1.0,
    }
    report = mixed_span(corpora, ["ko", "en"], [1, 1], config, spec)
    recipe = TrainingRecipe.from_dict(
        {"schema_version": 1, "stage": "base", "corruption": {"full_mask_prob": 1.0}},
        "base",
    )
    tasks, _ = base_stage(
        MixedCorpus(list(zip(corpora, [1, 1]))),
        config,
        2,
        seed=7,
        bucket=bucketed,
        recipe=recipe,
    )
    expected = {
        lang: {"rows": 0, "input_positions": 0, "canvas_positions": 0}
        for lang in ("ko", "en")
    }
    padded_input = padded_canvas = 0
    for _ in range(10):
        batch = tasks["span"].next_batch()
        padded_input += batch["input_ids"].numel()
        padded_canvas += batch["canvas"].numel()
        for row in range(len(batch["input_ids"])):
            language = "ko" if int(batch["input_ids"][row, 0]) == 8 else "en"
            expected[language]["rows"] += 1
            expected[language]["input_positions"] += int(
                batch["input_valid"][row].sum()
            )
            expected[language]["canvas_positions"] += int(
                batch["canvas_valid"][row].sum()
            )
    for language, values in expected.items():
        group = report["by_language"][language]
        assert all(group[key] == value for key, value in values.items())
        assert group["fully_masked_rows"] == group["rows"]
        assert group["supervised_targets"] == group["canvas_positions"]
    assert report["padded_input_positions"] == padded_input
    assert report["padded_canvas_positions"] == padded_canvas


def test_span_attribution_keeps_original_indices_after_filtering_and_bucketing():
    config = geometry()
    documents = [[8] * 5, [9], [10] * 80, [11] * 3, []]
    rng, generator = random.Random(5), torch.Generator().manual_seed(5)
    replay, other_generator = random.Random(5), torch.Generator().manual_seed(5)
    actual = build_span_batch(
        documents, config, rng, generator, bucket=(64, 32, 4), return_indices=True
    )
    expected = build_span_batch(
        documents, config, replay, other_generator, bucket=(64, 32, 4)
    )
    assert set(actual) - set(expected) == {"source_indices"}
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    assert rng.getstate() == replay.getstate() and torch.equal(
        generator.get_state(), other_generator.get_state()
    )
    for row, index in enumerate(actual["source_indices"].tolist()):
        assert index in (0, 2, 3)
        assert int(actual["input_ids"][row, 0]) == documents[index][0]


@pytest.mark.parametrize("probability", [-1, 1.1, float("nan"), True])
def test_invalid_full_canvas_probability(probability):
    with pytest.raises(ValueError, match="full_mask_prob"):
        validate(
            {
                "schema_version": 1,
                "model_config": "model.json",
                "full_mask_prob": probability,
                "corpora": [
                    {"language": "en", "path": "train", "weight": 1, "split": "train"}
                ],
            }
        )
