"""Private pair auditing and preparation use synthetic, isolated inputs."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

SPEC = importlib.util.spec_from_file_location(
    "pair_view_preparation_tests", Path(__file__).resolve().parents[1] / "tools/prepare_pair_views.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


@pytest.fixture
def workspace(tmp_path):
    repo, inputs = tmp_path / "repo", tmp_path / "originals"
    repo.mkdir()
    inputs.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / ".gitignore").write_text("/.private/\n/data/\n", encoding="utf-8")
    (repo / ".private").mkdir()
    words = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "[UNK]"] + [f"w{i}" for i in range(30)]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(words[:4])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[SEP] $A [EOS]", special_tokens=[("[SEP]", 3), ("[EOS]", 2)])
    tokenizer.save(str(inputs / "tokenizer.json"))
    write_rows(inputs / "train.jsonl", [pair("w0", "w1"), pair("w2", "w3")])
    write_rows(inputs / "eval.jsonl", [pair("w4", "w4")])
    return repo, inputs


def pair(query, positive):
    return {"query": query, "positive": positive}


def write_rows(path, rows):
    path.write_text("\n" + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def plan(workspace, job="v1", **changes):
    _, inputs = workspace
    value = {"job": job, "train": str(inputs / "train.jsonl"), "eval": str(inputs / "eval.jsonl"),
             "tokenizer": str(inputs / "tokenizer.json"), "max_input_len": 2}
    value.update(changes)
    return value


def policy(**changes):
    value = {"threshold": 0.8, "max_high_overlap_pairs": 2, "min_low_overlap_pairs": 2, "seed": 23}
    value.update(changes)
    return value


def private_files(workspace, job="v1"):
    return workspace[0] / ".private/pair-views" / job


def output_files(workspace, job="v1"):
    return workspace[0] / "data/prepared/pair-views" / job


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_rows(workspace, job):
    return [item["row"] for item in records(private_files(workspace, job) / "selection.jsonl")
            if item["split"] == "train" and item["selected"]]


def test_three_way_cap_keeps_validation_and_test_byte_identical(workspace):
    repo, inputs = workspace
    write_rows(inputs / "valid.jsonl", [pair("w4", "w4")])
    write_rows(inputs / "test.jsonl", [pair("w5", "w5")])
    value = plan(workspace, valid=str(inputs / "valid.jsonl"), test=str(inputs / "test.jsonl"), selection=policy())
    del value["eval"]
    report = tool.prepare_pairs(value, repo, prepare=True, progress=False)
    assert set(report["counts"]) == {"train", "valid", "test"}
    for split in ("valid", "test"):
        assert (output_files(workspace) / f"{split}.jsonl").read_bytes() == (inputs / f"{split}.jsonl").read_bytes()
        assert report["retained_pairs"][split] == 1
    assert report["verification"]["three_way_duplicates_absent"] is True


@pytest.mark.parametrize("duplicate_split", ["train", "valid", "test"])
def test_three_way_pair_audit_rejects_duplicate_content(workspace, duplicate_split):
    repo, inputs = workspace
    write_rows(inputs / "valid.jsonl", [pair("w4", "w4")])
    write_rows(inputs / "test.jsonl", [pair("w5", "w5")])
    value = plan(workspace, valid=str(inputs / "valid.jsonl"), test=str(inputs / "test.jsonl"))
    del value["eval"]
    path = inputs / f"{duplicate_split}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(pair("w0", "w1")) + "\n")
    with pytest.raises(tool.PreparationError, match="duplicate or overlapping"):
        tool.prepare_pairs(value, repo, progress=False)
    assert not (private_files(workspace) / "COMPLETE.json").exists()


def test_three_way_pair_environment_paths_are_explicit(workspace, monkeypatch, capsys):
    repo, inputs = workspace
    write_rows(inputs / "valid.jsonl", [pair("w4", "w4")])
    write_rows(inputs / "test.jsonl", [pair("w5", "w5")])
    for name, value in {"NURI_PAIR_VIEW_JOB": "env01", "NURI_PAIR_DATA": str(inputs / "train.jsonl"),
                        "NURI_VALID_PAIRS": str(inputs / "valid.jsonl"), "NURI_TEST_PAIRS": str(inputs / "test.jsonl"),
                        "NURI_TOKENIZER": str(inputs / "tokenizer.json"), "NURI_PAIR_MAX_INPUT_LEN": "2"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NURI_PAIR_VIEW_PLAN", raising=False)
    assert tool.main(["--repo", str(repo), "--no-progress"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "mode": "audit"}
    report = json.loads((private_files(workspace, "env01") / "report.json").read_text())
    assert set(report["counts"]) == {"train", "valid", "test"}


def test_audit_only_is_default_and_binds_private_evidence(workspace):
    repo, inputs = workspace
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in inputs.iterdir()}
    report = tool.prepare_pairs(plan(workspace, selection=policy()), repo, progress=False)
    assert report["mode"] == "audit" and report["retained_pairs"] is None and report["outputs"] == {}
    assert not output_files(workspace).exists()
    assert report["counts"]["train"]["pairs"] == 2 and report["counts"]["eval"]["pairs"] == 1
    assert report["counts"]["train"]["zero_overlap_pairs"] == 2
    assert all(item["selected"] is None for item in records(private_files(workspace) / "selection.jsonl"))
    evidence = json.loads((private_files(workspace) / "COMPLETE.json").read_text())
    assert evidence["report_sha256"] == hashlib.sha256((private_files(workspace) / "report.json").read_bytes()).hexdigest()
    for key in ("train", "eval", "tokenizer"):
        assert report["inputs"][key]["sha256"] == hashlib.sha256(Path(report["inputs"][key]["path"]).read_bytes()).hexdigest()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == snapshot for path, snapshot in before.items())


def test_overlap_uses_retrieval_prefix_before_excluding_specials(workspace):
    repo, inputs = workspace
    write_rows(inputs / "train.jsonl", [pair("w0 w1 w9 w9 w9", "w2 w1 w9 w9 w9"),
                                       pair("[EOS] [PAD] w0", "[EOS] [PAD] w1"),
                                       pair("w0", "w1"), pair("w2", "w2")])
    report = tool.prepare_pairs(plan(workspace), repo, progress=False)
    scores = [item["overlap"] for item in records(private_files(workspace) / "rows.jsonl") if item["split"] == "train"]
    assert scores == [pytest.approx(1 / 3), None, 0.0, 1.0]
    summary = report["counts"]["train"]
    assert summary["query_truncated_pairs"] == summary["positive_truncated_pairs"] == 2
    assert summary["zero_overlap_pairs"] == summary["empty_lexical_pairs"] == 1
    assert sum(bucket["pairs"] for bucket in summary["overlap_bins"]) == 2
    assert report["special_token_ids"] == [0, 1, 2, 3]


def test_explicit_extra_special_ids_and_empty_token_views(workspace):
    repo, inputs = workspace
    write_rows(inputs / "train.jsonl", [pair("w0", "w0"), pair("", "w1")])
    report = tool.prepare_pairs(plan(workspace, special_token_ids=[5]), repo, progress=False)
    assert report["counts"]["train"]["empty_lexical_pairs"] == 2
    assert report["counts"]["train"]["retrieval_eligible_pairs"] == 1
    assert report["counts"]["train"]["zero_overlap_pairs"] == 0


def test_deterministic_training_cap_preserves_low_rows_order_and_eval_bytes(workspace):
    repo, inputs = workspace
    original = [pair("w0", "w1"), pair("w2", "w3")] + [pair(f"w{i}", f"w{i}") for i in range(4, 20)]
    write_rows(inputs / "train.jsonl", original)
    # Different whitespace, CRLF and Unicode prove eval is copied rather than reserialized.
    evaluation = b'\r\n { "positive" : "w4", "query" : "w4" }\r\n' + \
        json.dumps(pair("한국어 w1", "한국어 w1"), ensure_ascii=False).encode("utf-8") + b'\r\n'
    (inputs / "eval.jsonl").write_bytes(evaluation)
    original_train = (inputs / "train.jsonl").read_bytes()
    for job, seed in (("a", 23), ("b", 23), ("c", 24)):
        report = tool.prepare_pairs(plan(workspace, job, selection=policy(seed=seed)), repo,
                                    prepare=True, progress=False)
        chosen = selected_rows(workspace, job)
        assert chosen[:2] == [1, 2] and chosen == sorted(chosen) and len(chosen) == 4
        assert records(output_files(workspace, job) / "train.jsonl") == [original[index - 1] for index in chosen]
        assert (output_files(workspace, job) / "eval.jsonl").read_bytes() == evaluation
        assert report["retained_pairs"] == {"train": 4, "eval": 2}
        assert report["verification"]["train_content_and_order"] == "passed"
        assert report["verification"]["eval_byte_identical"] is True
        assert set(json.loads((output_files(workspace, job) / "COMPLETE.json").read_text())) == {"status", "schema_version"}
        assert report["outputs"]["train"]["sha256"] == hashlib.sha256((output_files(workspace, job) / "train.jsonl").read_bytes()).hexdigest()
    assert selected_rows(workspace, "a") == selected_rows(workspace, "b")
    assert selected_rows(workspace, "a") != selected_rows(workspace, "c")
    assert (inputs / "train.jsonl").read_bytes() == original_train


@pytest.mark.parametrize("cap, retained", [(0, 2), (100, 5)])
def test_zero_and_nonbinding_caps(workspace, cap, retained):
    repo, inputs = workspace
    write_rows(inputs / "train.jsonl", [pair("w0", "w1"), pair("w2", "w3")] + [pair("w4", "w4")] * 3)
    report = tool.prepare_pairs(plan(workspace, selection=policy(max_high_overlap_pairs=cap)), repo,
                                prepare=True, progress=False)
    assert report["retained_pairs"]["train"] == retained


def test_empty_lexical_rows_never_satisfy_low_overlap_support(workspace):
    repo, inputs = workspace
    write_rows(inputs / "train.jsonl", [pair("[EOS]", "[EOS]"), pair("w4", "w4")])
    with pytest.raises(tool.PreparationError, match="Too few low-overlap"):
        tool.prepare_pairs(plan(workspace, selection=policy(min_low_overlap_pairs=1)), repo,
                           prepare=True, progress=False)
    assert (private_files(workspace) / "FAILED.json").exists()
    assert not (private_files(workspace) / "COMPLETE.json").exists()
    assert not (output_files(workspace) / "COMPLETE.json").exists()


def test_unknown_lexical_rows_are_preserved_without_low_support_credit(workspace):
    repo, inputs = workspace
    write_rows(inputs / "train.jsonl", [pair("w0", "w1"), pair("w2", "w3"), pair("[EOS]", "[EOS]")])
    report = tool.prepare_pairs(plan(workspace, selection=policy()), repo, prepare=True, progress=False)
    assert report["counts"]["train"]["low_overlap_pairs"] == 2
    assert report["retained_pairs"]["train"] == 3
    retained = report["retention_counts"]["train"]
    assert retained["undefined_overlap_retained"] == 1 and retained["excluded_pairs"] == 0


@pytest.mark.parametrize("changes", [
    {"job": "../public"}, {"job": "nested/job"}, {"max_input_len": 0}, {"max_input_len": True},
    {"output_dir": "public"}, {"report": "public.json"}, {"overlap_bins": [0, 0.5, 0.5, 1]},
    {"overlap_bins": [0, float("nan"), 1]}, {"special_token_ids": [-1]},
])
def test_invalid_plan_cannot_create_leaking_or_escaped_outputs(workspace, changes):
    repo, _ = workspace
    with pytest.raises(tool.PreparationError):
        tool.prepare_pairs(plan(workspace, **changes), repo, progress=False)
    assert not (repo / ".private/pair-views").exists() and not (repo / "data").exists()


@pytest.mark.parametrize("selection", [None, {"threshold": 0.8}, policy(threshold=float("inf")),
                                        policy(max_high_overlap_pairs=-1), policy(min_low_overlap_pairs=0)])
def test_preparation_needs_a_complete_explicit_policy(workspace, selection):
    repo, _ = workspace
    with pytest.raises(tool.PreparationError):
        tool.prepare_pairs(plan(workspace, selection=selection), repo, prepare=True, progress=False)


def test_expected_hash_mismatch_leaves_no_completion(workspace):
    repo, _ = workspace
    with pytest.raises(tool.PreparationError, match="expected SHA256"):
        tool.prepare_pairs(plan(workspace, expected_sha256={"train": "0" * 64}), repo, progress=False)
    assert not (private_files(workspace) / "COMPLETE.json").exists()


@pytest.mark.parametrize("invalid", [b'{"query":', b'\xff', b'{"query":"w0","positive":"w1","source":"private"}',
                                    b'{"query":null,"positive":"w1"}'])
def test_malformed_or_metadata_rows_fail_without_copying_eval(workspace, invalid):
    repo, inputs = workspace
    (inputs / "eval.jsonl").write_bytes(invalid)
    with pytest.raises(tool.PreparationError):
        tool.prepare_pairs(plan(workspace, selection=policy()), repo, prepare=True, progress=False)
    assert not (output_files(workspace) / "eval.jsonl").exists()
    assert not (output_files(workspace) / "COMPLETE.json").exists()


def test_existing_success_or_failure_is_never_overwritten(workspace):
    repo, _ = workspace
    config = plan(workspace, selection=policy())
    tool.prepare_pairs(config, repo, progress=False)
    before = (private_files(workspace) / "report.json").read_bytes()
    with pytest.raises(tool.PreparationError, match="already exists"):
        tool.prepare_pairs(config, repo, prepare=True, progress=False)
    assert (private_files(workspace) / "report.json").read_bytes() == before


def test_failed_job_is_never_reused(workspace):
    repo, _ = workspace
    config = plan(workspace, selection=policy(min_low_overlap_pairs=100))
    with pytest.raises(tool.PreparationError, match="Too few low-overlap"):
        tool.prepare_pairs(config, repo, prepare=True, progress=False)
    before = (private_files(workspace) / "FAILED.json").read_bytes()
    with pytest.raises(tool.PreparationError, match="already exists"):
        tool.prepare_pairs(config, repo, progress=False)
    assert (private_files(workspace) / "FAILED.json").read_bytes() == before


@pytest.mark.parametrize("stage", ["runtime", "private"])
@pytest.mark.parametrize("after_write", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_completion_write_failure_removes_both_success_markers(workspace, monkeypatch, stage, after_write, error_type):
    repo, _ = workspace
    actual_write = tool._write_json
    target = (output_files(workspace) if stage == "runtime" else private_files(workspace)) / "COMPLETE.json"

    def failing_write(path, value):
        if path == target:
            if after_write:
                actual_write(path, value)
            raise error_type("Injected final marker failure")
        actual_write(path, value)

    monkeypatch.setattr(tool, "_write_json", failing_write)
    with pytest.raises(tool.PreparationError, match="details are stored privately"):
        tool.prepare_pairs(plan(workspace, selection=policy()), repo, prepare=True, progress=False)
    assert (private_files(workspace) / "FAILED.json").exists()
    failure = json.loads((private_files(workspace) / "FAILED.json").read_text())
    assert failure["status"] == ("cancelled" if error_type is KeyboardInterrupt else "failed")
    assert error_type.__name__ in failure["error"]
    assert (private_files(workspace) / "report.json").exists()  # Evidence and runtime examples survive.
    assert (output_files(workspace) / "train.jsonl").exists()
    assert not (private_files(workspace) / "COMPLETE.json").exists()
    assert not (output_files(workspace) / "COMPLETE.json").exists()


def test_one_usable_pair_plus_empty_input_does_not_pass_training_support(workspace):
    repo, inputs = workspace
    write_rows(inputs / "train.jsonl", [pair("w0", "w1"), pair("", "w2")])
    with pytest.raises(tool.PreparationError, match="two retrieval-eligible pairs"):
        tool.prepare_pairs(plan(workspace, selection=policy(min_low_overlap_pairs=1)), repo,
                           prepare=True, progress=False)


def test_source_mutation_is_detected_before_completion(workspace, monkeypatch):
    repo, inputs = workspace
    actual_audit = tool._audit

    def changed_input(*args, **kwargs):
        result = actual_audit(*args, **kwargs)
        with (inputs / "train.jsonl").open("ab") as stream:
            stream.write(b'\n')
        return result

    monkeypatch.setattr(tool, "_audit", changed_input)
    with pytest.raises(tool.PreparationError, match="changed during the job"):
        tool.prepare_pairs(plan(workspace), repo, progress=False)
    assert not (private_files(workspace) / "COMPLETE.json").exists()


def test_unignored_destinations_are_rejected(workspace):
    repo, _ = workspace
    (repo / ".gitignore").write_text("", encoding="utf-8")
    with pytest.raises(tool.PreparationError, match="ignored by Git"):
        tool.prepare_pairs(plan(workspace), repo, progress=False)
    assert not private_files(workspace).exists()


def test_forced_tracked_destination_is_rejected(workspace):
    repo, _ = workspace
    destination = output_files(workspace)
    destination.mkdir(parents=True)
    existing = destination / "train.jsonl"
    existing.write_text("private", encoding="utf-8")
    subprocess.run(["git", "add", "-f", str(existing)], check=True, cwd=repo)
    with pytest.raises(tool.PreparationError, match="tracked files"):
        tool.prepare_pairs(plan(workspace), repo, progress=False)
    assert existing.read_text() == "private"


def test_plan_must_be_private_ignored_and_untracked(workspace, capsys):
    repo, _ = workspace
    public = repo / "public.json"
    public.write_text(json.dumps(plan(workspace)), encoding="utf-8")
    assert tool.main(["--repo", str(repo), "--plan", str(public), "--no-progress"]) == 1
    assert "permitted directory" in capsys.readouterr().err
    private = repo / ".private/plan.json"
    private.write_text(public.read_text(), encoding="utf-8")
    subprocess.run(["git", "add", "-f", str(private)], check=True, cwd=repo)
    assert tool.main(["--repo", str(repo), "--plan", str(private), "--no-progress"]) == 1
    assert "tracked files" in capsys.readouterr().err


def test_environment_audit_and_private_plan_precedence(workspace, monkeypatch, capsys):
    repo, inputs = workspace
    monkeypatch.delenv("NURI_PAIR_VIEW_PLAN", raising=False)
    for key, value in {"NURI_PAIR_VIEW_JOB": "env", "NURI_PAIR_DATA": inputs / "train.jsonl",
                       "NURI_EVAL_PAIRS": inputs / "eval.jsonl", "NURI_TOKENIZER": inputs / "tokenizer.json",
                       "NURI_PAIR_MAX_INPUT_LEN": 2}.items():
        monkeypatch.setenv(key, str(value))
    assert tool.main(["--repo", str(repo), "--no-progress"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "mode": "audit"}
    private = repo / ".private/plan.json"
    private.write_text(json.dumps(plan(workspace, "planned", selection=policy())), encoding="utf-8")
    monkeypatch.setenv("NURI_PAIR_DATA", "unused")
    assert tool.main(["--repo", str(repo), "--plan", str(private), "--job", "explicit",
                      "--prepare", "--no-progress"]) == 0
    result = capsys.readouterr()
    assert str(inputs) not in result.out + result.err
    assert (output_files(workspace, "explicit") / "COMPLETE.json").exists()


def test_redirected_destinations_are_rejected(workspace, tmp_path):
    repo, _ = workspace
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        (repo / "data").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows session cannot create a directory symlink")
    with pytest.raises(tool.PreparationError, match="redirected paths"):
        tool.prepare_pairs(plan(workspace), repo, progress=False)
    assert not list(outside.iterdir())
