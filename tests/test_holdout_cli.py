"""Explicit private split inputs, early overlap rejection and real CLI curves."""

import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.data import write_corpus
from scripts import train as cli


def inputs(tmp_path):
    words = [
        "[PAD]",
        "[MASK]",
        "[EOS]",
        "[SEP]",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "[UNK]",
    ]
    words += [f"word{i}" for i in range(48)]
    tokenizer = Tokenizer(
        models.WordLevel(
            {word: index for index, word in enumerate(words)}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    for shift, split in enumerate(("train", "valid", "test")):
        folder = tmp_path / split
        write_corpus(
            folder,
            [
                [8 + (shift * 11 + row + pos) % 48 for pos in range(12)]
                for row in range(10)
            ],
        )
        instruction = [
            {
                "messages": [{"role": "user", "content": f"word{row} word{shift}"}],
                "response": f"word{row + 10 + shift} word{row + 20}",
            }
            for row in range(8)
        ]
        pairs = [
            {
                "query": f"word{row} word{shift}",
                "positive": f"word{row + 10} word{shift}",
            }
            for row in range(8)
        ]
        (folder / "instructions.jsonl").write_text(
            "\n".join(map(json.dumps, instruction)), encoding="utf-8"
        )
        (folder / "pairs.jsonl").write_text(
            "\n".join(map(json.dumps, pairs)), encoding="utf-8"
        )
    data = {"schema_version": 1, "valid_every": 2, "batch_size": 2}
    for split in ("valid", "test"):
        data[split] = {
            "corpus": [split],
            "instructions": f"{split}/instructions.jsonl",
            "pairs": f"{split}/pairs.jsonl",
            "batches_per_task": 2,
            "seed": 9,
            "depth": 2,
            "exposure_caveats": ["retrospective partition"],
        }
    (tmp_path / "holdout.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def args(tmp_path, stage="base", extra=()):
    common = [
        "--stage",
        stage,
        "--out",
        str(tmp_path / stage),
        "--steps",
        "3",
        "--batch-size",
        "2",
        "--holdout-config",
        str(tmp_path / "holdout.json"),
        "--no-progress",
        "--log-every",
        "1",
        *extra,
    ]
    if stage == "base":
        return [
            *common,
            "--preset",
            "tiny",
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--corpus",
            str(tmp_path / "train"),
        ]
    return (
        [
            *common,
            "--init",
            str(tmp_path / "base/final"),
            "--instructions",
            str(tmp_path / "train/instructions.jsonl"),
            "--reasoning-examples",
            "2",
        ]
        if stage == "it"
        else [
            *common,
            "--init",
            str(tmp_path / "base/final"),
            "--pairs",
            str(tmp_path / "train/pairs.jsonl"),
        ]
    )


def test_real_base_valid_test_curves_counts_and_fixed_digests(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    inputs(tmp_path)
    cli.main(args(tmp_path, extra=["--tensorboard"]))
    reports = [
        json.loads(line)
        for line in (tmp_path / "base/holdout_log.jsonl").read_text().splitlines()
    ]
    assert [(report["split"], report["step"]) for report in reports] == [
        ("valid", 2),
        ("valid", 3),
        ("test", 3),
    ]
    assert (
        reports[0]["tasks"]["mlm"]["batch_sha256"]
        == reports[1]["tasks"]["mlm"]["batch_sha256"]
    )
    assert reports[-1]["metadata"]["exposure_caveats"] == ["retrospective partition"]
    assert reports[-1]["unassessed_training_tasks"] == []
    events = EventAccumulator(
        str(tmp_path / "base/tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    assert [point.step for point in events.Scalars("valid/mlm/objective_loss")] == [
        2,
        3,
    ]
    assert [point.step for point in events.Scalars("test/span/objective_loss")] == [3]
    allowed_text = {
        f"{split}/{name}/text_summary"
        for split in ("valid", "test")
        for name in (
            "coverage",
            "policy_sha256",
            "mlm/batch_sha256",
            "mlm/target_sha256",
            "span/batch_sha256",
            "span/target_sha256",
        )
    }
    assert set(events.Tags()["tensors"]) == allowed_text
    assert (tmp_path / "base/final/model.safetensors").exists()


def test_it_missing_explicit_reasoning_is_reported_and_pair_views_count_both(tmp_path):
    inputs(tmp_path)
    cli.main(args(tmp_path))
    cli.main(args(tmp_path, "it"))
    reports = [
        json.loads(line)
        for line in (tmp_path / "it/holdout_log.jsonl").read_text().splitlines()
    ]
    assert reports[-1]["unassessed_training_tasks"] == ["reasoning"]
    assert set(reports[-1]["tasks"]["instruction"]["metrics"]) == {
        "loss",
        "denoise",
        "length",
    }
    assert reports[-1]["tasks"]["instruction"]["source_pool"]["selected_pool_rows"] == 4
    cli.main(args(tmp_path, "embed"))
    report = json.loads(
        (tmp_path / "embed/holdout_log.jsonl").read_text().splitlines()[-1]
    )
    pair = report["tasks"]["pairs"]
    assert pair["counts"]["rows"] == 4
    assert pair["candidate_rows_per_batch"] == [2, 2]
    assert pair["counts"]["valid_input_tokens"] == 16


@pytest.mark.parametrize("flags", [["--valid-every", "1"], ["--valid-every", "-1"]])
def test_invalid_validation_flags_fail_parse_before_model_allocation(flags):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(["--stage", "base", "--out", "unused", "--steps", "1", *flags])
    assert error.value.code == 2


def test_overlap_or_missing_split_fails_before_allocation_and_no_fallback(
    tmp_path, monkeypatch
):
    data = inputs(tmp_path)
    data["test"]["corpus"] = ["train"]
    (tmp_path / "holdout.json").write_text(json.dumps(data))
    monkeypatch.setattr(
        cli, "NuriModel", lambda *args: pytest.fail("allocated before path rejection")
    )
    with pytest.raises(ValueError, match="distinct resolved"):
        cli.main(args(tmp_path))
    del data["test"]["corpus"]
    (tmp_path / "holdout.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="own stage inputs"):
        cli.main(args(tmp_path))


def test_manifest_binds_content_without_confusing_equal_offsets_with_duplicate_corpora(
    tmp_path,
):
    data = inputs(tmp_path)
    files = []
    for split in ("train", "valid", "test"):
        for name in ("tokens.bin", "offsets.npy", "instructions.jsonl", "pairs.jsonl"):
            path = tmp_path / split / name
            files.append(
                {
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"files": files}))
    data.update(
        input_manifest="inputs.json",
        input_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    (tmp_path / "holdout.json").write_text(json.dumps(data))
    assert cli.load_holdout_config(cli.parse_args(args(tmp_path))) is not None
    # Different paths with the same full corpus content are a real overlap.
    for name in ("tokens.bin", "offsets.npy"):
        next(entry for entry in files if entry["path"] == f"test/{name}")["sha256"] = (
            next(entry["sha256"] for entry in files if entry["path"] == f"train/{name}")
        )
    manifest.write_text(json.dumps({"files": files}))
    data["input_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "holdout.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical content"):
        cli.load_holdout_config(cli.parse_args(args(tmp_path)))


def test_environment_and_explicit_validation_interval_precedence(tmp_path, monkeypatch):
    inputs(tmp_path)
    monkeypatch.setenv("NURI_HOLDOUT_CONFIG", str(tmp_path / "holdout.json"))
    arguments = [
        "--stage",
        "base",
        "--out",
        "unused",
        "--steps",
        "1",
        "--valid-every",
        "0",
        "--corpus",
        str(tmp_path / "train"),
    ]
    assert cli.load_holdout_config(cli.parse_args(arguments))["valid_every"] == 0
