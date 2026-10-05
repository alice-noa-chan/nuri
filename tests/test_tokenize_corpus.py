import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.data import TokenCorpus


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "tokenize_corpus.py"
    spec = importlib.util.spec_from_file_location("tokenize_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenize_text_and_jsonl(tmp_path):
    vocab = {"[PAD]": 0, "[MASK]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4, "world": 5, "again": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "a.txt").write_text("hello world\nhello\n\nworld again hello\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(json.dumps({"text": "again world"}) + "\n", encoding="utf-8")

    load_script().main([str(tmp_path / "a.txt"), str(tmp_path / "b.jsonl"), "--tokenizer", str(tmp_path),
                        "--out", str(tmp_path / "corpus"), "--batch-size", "2"])
    corpus = TokenCorpus(tmp_path / "corpus")
    assert [corpus.document(i) for i in range(len(corpus))] == [[4, 5], [5, 6, 4], [6, 5]]
    stats = json.loads((tmp_path / "corpus" / "stats.json").read_text())
    assert stats == {"documents_seen": 4, "tokens": 7, "documents_written": 3}


def split_plan(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / ".gitignore").write_text("/data/\n/.private/\n")
    (repo / ".private").mkdir()
    (repo / "data/prepared").mkdir(parents=True)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "again": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_file = repo / "data/tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    manifest = {"status": "complete", "split_policy": {"mode": "preserve_legacy_train", "valid_share": 0.5},
                "splits": {}, "files": []}
    for split, text in (("train", "hello world"), ("valid", "world again"), ("test", "again hello")):
        path = repo / f"data/prepared/{split}.jsonl"
        path.write_text(json.dumps({"text": text}) + "\n")
        entry = {"path": str(path), "rows": 1, "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["splits"][split] = {"text": {"en": entry}}
        manifest["files"].append({k: entry[k] for k in ("path", "bytes", "sha256")})
    preparation = repo / ".private/prepared.json"
    preparation.write_text(json.dumps(manifest))
    plan = {"schema_version": 1, "job": "split01", "tokenizer": str(tokenizer_file),
            "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(), "vocab_size": 4,
            "preparation_manifest": str(preparation),
            "preparation_manifest_sha256": hashlib.sha256(preparation.read_bytes()).hexdigest(), "batch_size": 2}
    path = repo / ".private/tokenize.json"
    path.write_text(json.dumps(plan))
    return repo, path, plan


def test_private_three_way_tokenization_and_bound_train_reuse(tmp_path):
    repo, path, plan = split_plan(tmp_path)
    script = load_script()
    result = script.tokenize_splits(str(path), str(repo), False)
    assert result["status"] == "complete" and len(result["files"]) == 9
    for split, expected in (("train", [1, 2]), ("valid", [2, 3]), ("test", [3, 1])):
        corpus = TokenCorpus(repo / f"data/tokens/split01/en/{split}")
        assert corpus.document(0) == expected
    old = result["splits"]["train"]["text"]["en"]
    plan.update(job="split02", reuse_train={"en": {"path": old["corpus"]["path"],
                "prepared_sha256": old["sha256"], "stats": old["corpus"]["stats"],
                "files": {name: entry["sha256"] for name, entry in old["corpus"]["files"].items()}}})
    path.write_text(json.dumps(plan))
    reused = script.tokenize_splits(str(path), str(repo), False)
    assert reused["splits"]["train"]["text"]["en"]["corpus"]["reused"] is True
    assert not (repo / "data/tokens/split02/en/train").exists()
    assert TokenCorpus(repo / "data/tokens/split02/en/test").document(0) == [3, 1]


@pytest.mark.parametrize("failure", ["vocab", "manifest", "tokenizer", "outside_private", "changed_input"])
def test_split_tokenization_rejects_bad_bindings(tmp_path, failure):
    repo, path, plan = split_plan(tmp_path)
    if failure == "vocab":
        plan["vocab_size"] = 5
    elif failure == "manifest":
        plan["preparation_manifest_sha256"] = "0" * 64
    elif failure == "tokenizer":
        plan["tokenizer_sha256"] = "0" * 64
    elif failure == "outside_private":
        plan["preparation_manifest"] = str(repo / "data/prepared/train.jsonl")
    else:
        (repo / "data/prepared/test.jsonl").write_text('{"text":"changed text"}\n')
    path.write_text(json.dumps(plan))
    script = load_script()
    with pytest.raises(script.PreparationError):
        script.tokenize_splits(str(path), str(repo), False)
    assert not (repo / "data/tokens/split01/COMPLETE.json").exists()


def test_split_tokenization_interrupt_after_marker_removes_success(tmp_path, monkeypatch):
    repo, path, plan = split_plan(tmp_path)
    script = load_script()
    write = script._write_json

    def interrupted(path, value):
        write(path, value)
        if path.name == "COMPLETE.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(script, "_write_json", interrupted)
    with pytest.raises(script.PreparationError, match="partial outputs"):
        script.tokenize_splits(str(path), str(repo), False)
    assert not (repo / "data/tokens/split01/COMPLETE.json").exists()
    assert not (repo / ".private/tokenization-consumption-split01.json").exists()
    assert (repo / "data/tokens/split01/FAILED.json").exists()


@pytest.mark.parametrize("wrong_field,wrong_value", [("vocab_size", 3), ("tokenizer_sha256_at_load", "0" * 64)])
def test_reused_training_corpus_rejects_historical_tokenizer_marker(tmp_path, wrong_field, wrong_value):
    repo, path, plan = split_plan(tmp_path)
    script = load_script()
    result = script.tokenize_splits(str(path), str(repo), False)
    old = result["splits"]["train"]["text"]["en"]
    plan.update(job="split02", reuse_train={"en": {"path": old["corpus"]["path"],
        "prepared_sha256": old["sha256"], "stats": old["corpus"]["stats"],
        "files": {name: entry["sha256"] for name, entry in old["corpus"]["files"].items()}}})
    path.write_text(json.dumps(plan))
    marker = repo / "data/tokens/split01/COMPLETE.json"
    content = json.loads(marker.read_text())
    content[wrong_field] = wrong_value
    marker.write_text(json.dumps(content))
    with pytest.raises(script.PreparationError, match="exact loaded tokenizer"):
        script.tokenize_splits(str(path), str(repo), False)
    assert not (repo / "data/tokens/split02/COMPLETE.json").exists()
