"""Separate language objectives without changing training or split boundaries."""

import copy
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri import trainer
from nuri.checkpoint import load_pretrained
from nuri.config import tiny_config
from nuri.data import write_corpus
from nuri.monitoring import TrainingMonitor
from scripts import train as cli


def prepare_inputs(tmp_path):
    words = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"]
    words += [f"word{i}" for i in range(48)]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    data = {"schema_version": 1, "valid_every": 1, "batch_size": 2}
    for index, split in enumerate(("train", "valid", "test")):
        folder = tmp_path / split
        write_corpus(folder, [[8 + (index * 13 + row + pos) % 48 for pos in range(12)] for row in range(8)])
        rows = [{"messages": [{"role": "user", "content": f"word{row} word{index}"}],
                 "response": f"word{row + 12} word{row + 22}"} for row in range(6)]
        pairs = [{"query": f"word{row} word{index}", "positive": f"word{row + 12} word{index}"}
                 for row in range(6)]
        (folder / "instructions.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        (folder / "pairs.jsonl").write_text("\n".join(map(json.dumps, pairs)), encoding="utf-8")
        if split != "train":
            data[split] = {"corpus": [split], "instructions": f"{split}/instructions.jsonl",
                           "pairs": f"{split}/pairs.jsonl", "batches_per_task": 1, "seed": 9, "depth": 2}
    return data


def save_config(tmp_path, data):
    (tmp_path / "holdout.json").write_text(json.dumps(data), encoding="utf-8")


def arguments(tmp_path, out="base", stage="base"):
    common = ["--stage", stage, "--out", str(tmp_path / out), "--steps", "2", "--batch-size", "2",
              "--holdout-config", str(tmp_path / "holdout.json"), "--no-progress", "--log-every", "1"]
    if stage == "base":
        return common + ["--preset", "tiny", "--tokenizer", str(tmp_path / "tokenizer.json"),
                         "--corpus", str(tmp_path / "train")]
    if stage == "it":
        return common + ["--init", str(tmp_path / "base/final"), "--reasoning-examples", "2",
                         "--instructions", str(tmp_path / "train/instructions.jsonl")]
    return common + ["--init", str(tmp_path / "base/final"), "--pairs", str(tmp_path / "train/pairs.jsonl")]


def add_groups(data):
    # Synthetic aliases are allowed within a split; labels do not prove language coverage.
    for split in ("valid", "test"):
        resources = {key: value for key, value in data[split].items() if key in {"corpus", "instructions", "pairs"}}
        data[split]["language_groups"] = {language: copy.deepcopy(resources) for language in ("ko", "en", "ja")}


def reports(tmp_path, out):
    return [json.loads(line) for line in (tmp_path / out / "holdout_log.jsonl").read_text().splitlines()]


def assert_same_state(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same_state(a, b)
    else:
        assert left == right


def test_actual_language_curves_keep_aggregate_masks_weights_and_receipts(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    data = prepare_inputs(tmp_path)
    save_config(tmp_path, data)
    cli.main(arguments(tmp_path, "plain"))
    plain_reports = reports(tmp_path, "plain")
    add_groups(data)
    save_config(tmp_path, data)
    cli.main(arguments(tmp_path, "grouped") + ["--tensorboard"])
    grouped_reports = reports(tmp_path, "grouped")
    before = load_file(str(tmp_path / "plain/final/model.safetensors"))
    after = load_file(str(tmp_path / "grouped/final/model.safetensors"))
    assert before.keys() == after.keys()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    plain_state = torch.load(tmp_path / "plain/training_state/final.pt", map_location="cpu", weights_only=False)
    grouped_state = torch.load(tmp_path / "grouped/training_state/final.pt", map_location="cpu", weights_only=False)
    assert_same_state(plain_state["optimizer"], grouped_state["optimizer"])
    assert_same_state(plain_state["extra"]["random"], grouped_state["extra"]["random"])
    for plain, grouped in zip(plain_reports, grouped_reports, strict=True):
        assert (plain["split"], plain["step"]) == (grouped["split"], grouped["step"])
        for name in ("mlm", "span"):
            assert plain["tasks"][name] == grouped["tasks"][name]
        assert grouped["assessed_batches"] == 8
        assert len(grouped["tasks"]) == 8
        assert grouped["policy_sha256"] != plain["policy_sha256"]
    for name in grouped_reports[0]["tasks"]:
        assert grouped_reports[0]["tasks"][name]["batch_sha256"] == grouped_reports[1]["tasks"][name]["batch_sha256"]
    events = EventAccumulator(str(tmp_path / "grouped/tensorboard"), size_guidance={"scalars": 0}).Reload()
    for language in ("ko", "en", "ja"):
        for kind in ("mlm", "span"):
            assert [p.step for p in events.Scalars(f"valid/{language}/{kind}/objective_loss")] == [1, 2]
            assert [p.step for p in events.Scalars(f"test/{language}/{kind}/objective_loss")] == [2]
    assert len(events.Scalars("valid/mlm/objective_loss")) == 2
    args = cli.parse_args(arguments(tmp_path, "grouped"))
    model = load_pretrained(tmp_path / "grouped/final")
    config = cli.load_holdout_config(args)
    valid, _ = cli.build_holdouts(config, args, model.config, None)
    settings = trainer.TrainSettings(str(tmp_path / "grouped"), 2)
    assert not trainer._assessment_recorded(tmp_path / "plain/holdout/valid-step2.json", "valid", 2,
                                            valid, model, settings)


@pytest.mark.parametrize("failure", ["empty", "unknown", "list", "extra", "missing", "mismatch", "train_alias", "test_alias"])
def test_invalid_group_inputs_fail_before_allocating_model(tmp_path, monkeypatch, failure):
    data = prepare_inputs(tmp_path)
    add_groups(data)
    if failure == "empty":
        data["valid"]["language_groups"] = {}
    elif failure == "unknown":
        data["valid"]["language_groups"]["other"] = {"corpus": ["valid"]}
    elif failure == "list":
        data["valid"]["language_groups"] = []
    elif failure == "extra":
        data["valid"]["language_groups"]["en"]["seed"] = 1
    elif failure == "missing":
        del data["valid"]["language_groups"]["en"]["corpus"]
    elif failure == "mismatch":
        del data["test"]["language_groups"]["ja"]
    elif failure == "train_alias":
        data["valid"]["language_groups"]["en"]["corpus"] = ["train"]
    else:
        data["valid"]["language_groups"]["en"]["corpus"] = ["test"]
    save_config(tmp_path, data)
    monkeypatch.setattr(cli, "NuriModel", lambda *a, **k: pytest.fail("allocated before rejection"))
    with pytest.raises(ValueError):
        cli.main(arguments(tmp_path))


def test_manifest_must_cover_nested_language_resources(tmp_path):
    data = prepare_inputs(tmp_path)
    add_groups(data)
    write_corpus(tmp_path / "valid-en", [[8 + pos for pos in range(10)]])
    data["valid"]["language_groups"]["en"]["corpus"] = ["valid-en"]
    files = []
    for split in ("train", "valid", "test"):
        for name in ("tokens.bin", "offsets.npy", "instructions.jsonl", "pairs.jsonl"):
            path = tmp_path / split / name
            files.append({"path": path.relative_to(tmp_path).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    data.update(input_manifest="inputs.json", input_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    save_config(tmp_path, data)
    with pytest.raises(ValueError, match="cover all selected"):
        cli.load_holdout_config(cli.parse_args(arguments(tmp_path)))
    for name in ("tokens.bin", "offsets.npy"):
        path = tmp_path / "valid-en" / name
        files.append({"path": f"valid-en/{name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    data["input_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    save_config(tmp_path, data)
    assert cli.load_holdout_config(cli.parse_args(arguments(tmp_path))) is not None


def test_it_and_embed_groups_keep_separate_objectives_and_counts(tmp_path):
    data = prepare_inputs(tmp_path)
    save_config(tmp_path, data)
    cli.main(arguments(tmp_path))
    add_groups(data)
    for split in ("valid", "test"):
        data[split]["language_groups"]["ko"]["reasoning"] = f"{split}/instructions.jsonl"
    save_config(tmp_path, data)
    for stage in ("it", "embed"):
        cli.main(arguments(tmp_path, stage, stage))
        result = reports(tmp_path, stage)[-1]
        for language in ("ko", "en", "ja"):
            task = result["tasks"][f"{language}/{'instruction' if stage == 'it' else 'pairs'}"]
            assert task["counts"]["rows"] == 2
            assert task["source_pool"]["selected_pool_rows"] == 2
        if stage == "it":
            assert result["tasks"]["ko/reasoning"]["kind"] == "instruction"
        else:
            assert result["tasks"]["en/pairs"]["candidate_rows_per_batch"] == [2]


def test_only_fixed_language_and_objective_names_enter_tensorboard(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    task = {"kind": "instruction", "metrics": {"loss": 0.5}, "loss_numerators": {},
            "loss_denominators": {}, "counts": {}, "batches": 1, "batch_sha256": "a" * 64,
            "target_sha256": "b" * 64}
    report = {"split": "valid", "step": 1, "assessed_rows": 2, "assessed_batches": 2,
              "seconds": 0.1, "seed": 0, "depth": 4, "coverage": "bounded_sample",
              "policy_sha256": "c" * 64, "tasks": {"ko/reasoning": task, "private-label/instruction": task}}
    with TrainingMonitor(1, 0, 0, progress=False, tensorboard_dir=tmp_path) as monitor:
        monitor.record_holdout(report)
    tags = EventAccumulator(str(tmp_path)).Reload().Tags()
    assert "valid/ko/reasoning/objective_loss" in tags["scalars"]
    assert "valid/instruction/objective_loss" in tags["scalars"]
    assert all("private-label" not in tag for category in ("scalars", "tensors") for tag in tags[category])


@pytest.mark.parametrize("grouped", [False, True])
def test_explicit_reasoning_with_no_eligible_rows_is_not_silently_omitted(tmp_path, grouped):
    data = prepare_inputs(tmp_path)
    empty = tmp_path / "empty-reasoning.jsonl"
    empty.write_text("", encoding="utf-8")
    if grouped:
        add_groups(data)
        data["valid"]["reasoning"] = "valid/instructions.jsonl"
        data["valid"]["language_groups"]["ko"]["reasoning"] = empty.name
    else:
        data["valid"]["reasoning"] = empty.name
    save_config(tmp_path, data)
    args = cli.parse_args(arguments(tmp_path, "it", "it"))
    config = cli.load_holdout_config(args)
    tokenizer = cli.load_tokenizer(tmp_path / "tokenizer.json")
    with pytest.raises(ValueError, match="reasoning input has no eligible rows"):
        cli.build_holdouts(config, args, tiny_config(), tokenizer)
