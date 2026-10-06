"""Check source attribution against the actual builders, using synthetic TRAIN."""

import json
import random

import pytest
import torch

from nuri.config import NuriConfig
from nuri.data import TokenCorpus, build_mlm_batch, build_packed_mlm_batch, write_corpus
from tools.audit_batch_policy import main, mixed_mlm, validate


def geometry():
    return NuriConfig(vocab_size=64, hidden_size=16, num_heads=2, ffn_size=32, input_layers=1,
                      readout_layers=1, latent_slots=2, max_input_len=32, max_response_len=16,
                      embed_dim=8, train_depths=(1,))


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
        batch = (build_packed_mlm_batch(draw, 2, config, rng, generator) if packed
                 else build_mlm_batch([draw(), draw()], config, rng, generator))
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
    report = mixed_mlm(corpora, ["ko", "en"], [1, 1], geometry(), {"seed": 3, "batches": 8})
    groups = report["by_language"]
    assert all(group["input_positions"] > 0 and group["supervised_targets"] > 0 for group in groups.values())
    assert sum(group["input_positions"] for group in groups.values()) == 16 * 21
    assert all(group["rows"] is None for group in groups.values())


def test_cli_binds_inputs_and_preserves_existing_receipt(tmp_path):
    tmp_path = tmp_path / ".private"
    tmp_path.mkdir()
    config = geometry()
    config.save(tmp_path / "model.json")
    write_corpus(tmp_path / "train", [[8] * 12, [9] * 40])
    spec = {"schema_version": 1, "model_config": "model.json", "batches": 2,
            "corpora": [{"language": "ko", "path": "train", "weight": 1, "split": "train"}]}
    path, output = tmp_path / "audit.json", tmp_path / "receipt.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    main([str(path), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["optimizer_updates"] == 0 and report["gpu_used"] is False and report["test_used"] is False
    assert len(report["input_sha256"]) == 4
    assert report["span_isolated_by_language"]["ko"]["supervised_targets"] > 0
    prior = output.read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        main([str(path), "--output", str(output)])
    assert output.read_bytes() == prior


def test_cli_refuses_public_configuration_or_receipt(tmp_path):
    with pytest.raises(ValueError, match=".private"):
        main([str(tmp_path / "audit.json"), "--output", str(tmp_path / "receipt.json")])


@pytest.mark.parametrize("change", [{"batches": True}, {"batches": 513}, {"batch_size": 0},
                                    {"mlm_rate": float("nan")}, {"min_span": 0}, {"unknown": 1}])
def test_invalid_work_limits_rejected(change):
    spec = {"schema_version": 1, "model_config": "config.json",
            "corpora": [{"language": "en", "path": "train", "weight": 1, "split": "train"}], **change}
    with pytest.raises(ValueError):
        validate(spec)


def test_nontraining_resources_rejected():
    spec = {"schema_version": 1, "model_config": "config.json",
            "corpora": [{"language": "en", "path": "heldout", "weight": 1, "split": "test"}]}
    with pytest.raises(ValueError, match="split=train"):
        validate(spec)
