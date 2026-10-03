"""Private-plan preparation with fake inputs and isolated repository directories."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "nuri_private_preparation_tests", Path(__file__).resolve().parents[1] / "tools/prepare_data.py")
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


@pytest.fixture
def workspace(tmp_path):
    repo, source = tmp_path / "repo", tmp_path / "originals"
    (repo / ".private").mkdir(parents=True)
    source.mkdir()
    return repo, source


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def plan_file(workspace, inputs, **options):
    repo, source = workspace
    plan = {"job": "job01", "allowed_source_roots": [str(source)], "seed": 7,
            "eval_fraction": 0.0, "inputs": inputs}
    plan.update(options)
    path = repo / ".private/plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def item(path, identifier="input01", kind="text", **options):
    value = {"id": identifier, "path": str(path), "kind": kind, "lang": "en", "split": "train"}
    value.update(options)
    return value


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(workspace, path):
    return prepare_data.prepare(path, workspace[0], progress=False)


def test_import_is_an_independent_lossless_copy_with_private_metadata(workspace):
    repo, source = workspace
    original = source / "local.jsonl"
    text = "  first paragraph\n한국어 second paragraph  "
    jsonl(original, [{"payload": {"body": text}, "source_identity": "fake-local-origin", "metadata": {"private": True}}])
    before = original.read_bytes(), original.stat().st_mtime_ns, original.stat().st_size
    path = plan_file(workspace, [item(original, fields={"text": "payload.body"},
                                      sha256=hashlib.sha256(before[0]).hexdigest())], max_chars=9)
    counts = prepare(workspace, path)
    output = records(repo / "data/prepared/job01/text/en/train.jsonl")
    assert "".join(row["text"] for row in output) == text
    assert all(set(row) == {"text", "lang"} and len(row["text"]) <= 9 for row in output)
    assert counts["text"]["en"]["train"] == len(output)
    imported = repo / "data/raw/input01.jsonl"
    assert imported.read_bytes() == before[0]
    assert imported.stat().st_mtime_ns == before[1]
    assert imported.stat().st_ino != original.stat().st_ino
    assert not imported.is_symlink() and imported.stat().st_nlink == 1
    assert (original.read_bytes(), original.stat().st_mtime_ns, original.stat().st_size) == before
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    assert report["inputs"][0]["sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert report["inputs"][0]["size"] == before[2]
    assert report["inputs"][0]["accepted"] == 1
    assert (repo / "data/prepared/job01/COMPLETE.json").is_file()
    assert not (repo / "data/prepared/job01/.dedup.sqlite").exists()
    assert not list((repo / "data").rglob("*.part"))


def test_instruction_adapters_strip_metadata_and_validate_messages(workspace):
    repo, source = workspace
    explicit, extracted, prompt = (source / name for name in ("explicit.jsonl", "chat.jsonl", "prompt.jsonl"))
    jsonl(explicit, [
        {"messages": [{"role": "user", "content": "Question one", "origin": "fake"}], "answer": "Answer one", "origin": "fake"},
        {"messages": [{"role": "user", "content": 3}], "answer": "Invalid"},
        {"messages": [{"role": "unknown", "content": "Invalid"}], "answer": "Invalid"},
    ])
    jsonl(extracted, [{"messages": [{"role": "system", "content": "Be concise"},
                                     {"role": "user", "content": "Question two"},
                                     {"role": "assistant", "content": "Answer two", "extra": 1}]}])
    jsonl(prompt, [{"question": "Question three", "answer": "Answer three"}])
    path = plan_file(workspace, [
        item(explicit, "i01", "instruction", fields={"response": "answer"}),
        item(extracted, "i02", "instruction"),
        item(prompt, "i03", "instruction", fields={"prompt": "question", "response": "answer"}),
    ])
    prepare(workspace, path)
    output = records(repo / "data/prepared/job01/instructions/train.jsonl")
    assert [row["response"] for row in output] == ["Answer one", "Answer two", "Answer three"]
    assert all(set(row) == {"messages", "response"} for row in output)
    assert all(set(message) == {"role", "content"} for row in output for message in row["messages"])
    assert output[1]["messages"][-1]["role"] == "user"
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    assert report["inputs"][0]["accepted"] == 1
    assert report["inputs"][0]["dropped"] == 2


def test_explicit_answer_is_removed_from_the_observed_chat_prompt(workspace):
    from nuri.data import chat_prompt_text

    repo, source = workspace
    original = source / "chat.jsonl"
    prompt = [{"role": "system", "content": "Reply briefly"},
              {"role": "user", "content": "Earlier question"},
              {"role": "assistant", "content": "Earlier answer"},
              {"role": "user", "content": "Current question"}]
    response = "UNIQUE-CURRENT-TARGET"
    jsonl(original, [{"messages": prompt + [{"role": "assistant", "content": response}],
                      "answer": response}])
    prepare(workspace, plan_file(workspace, [item(original, kind="instruction", fields={"response": "answer"})]))
    row, = records(repo / "data/prepared/job01/instructions/train.jsonl")
    assert row["response"] == response
    assert row["messages"] == prompt
    assert response not in chat_prompt_text(row["messages"])
    assert "Earlier answer" in chat_prompt_text(row["messages"])


@pytest.mark.parametrize("messages,response", [
    ([{"role": "user", "content": "Question"}, {"role": "tool", "content": "Tool output"}], "Target"),
    ([{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Different target"}], "Target"),
    ([{"role": "system", "content": "System only"}], "Target"),
    ([{"role": "assistant", "content": "Target"}], "Target"),
    ([{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Earlier answer"},
      {"role": "assistant", "content": "Target"}], None),
])
def test_instruction_inputs_reject_unsupported_or_ambiguous_prompt_turns(workspace, messages, response):
    repo, source = workspace
    original = source / "invalid-chat.jsonl"
    row = {"messages": messages}
    if response is not None:
        row["response"] = response
    jsonl(original, [row])
    prepare(workspace, plan_file(workspace, [item(original, kind="instruction")]))
    assert not records(repo / "data/prepared/job01/instructions/train.jsonl")
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    assert report["inputs"][0]["accepted"] == 0
    assert report["inputs"][0]["dropped"] == 1


def test_tsv_pairs_use_mappings_and_positive_filter(workspace):
    repo, source = workspace
    original = source / "pairs.tsv"
    original.write_text("question\tdocument\tflag\tprovenance\n"
                        "Query one\tDocument one\t1\tfake\n"
                        "Query two\tDocument two\t0\tfake\n"
                        "Query three\t\t1\tfake\n", encoding="utf-8")
    path = plan_file(workspace, [item(original, kind="pairs", positive_values=["1"],
                                    fields={"query": "question", "positive": "document", "label": "flag"})])
    prepare(workspace, path)
    assert records(repo / "data/prepared/job01/pairs/train.jsonl") == [
        {"query": "Query one", "positive": "Document one"}]
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    assert report["inputs"][0]["accepted"] == 1
    assert report["inputs"][0]["dropped"] == 2


def test_group_holdout_is_deterministic_before_chunking(workspace):
    repo, source = workspace
    original = source / "groups.jsonl"
    rows = [{"text": "".join(f"G{group:02}D{index}P{part}xy" for part in range(5)), "family": group}
            for group in range(30) for index in range(2)]
    jsonl(original, rows)
    path = plan_file(workspace, [item(original, namespace="shared-group", group_field="family")],
                     eval_fraction=0.5, max_chars=9)
    prepare(workspace, path)

    def split_contents(job):
        return {split: records(repo / f"data/prepared/{job}/text/en/{split}.jsonl")
                for split in ("train", "eval")}

    first = split_contents("job01")
    assert first["train"] and first["eval"]
    for group in range(30):
        appearances = {split for split, output in first.items()
                       if any(row["text"].startswith(f"G{group:02}D") for row in output)}
        assert len(appearances) == 1
    path = plan_file(workspace, [item(original, namespace="shared-group", group_field="family")],
                     job="job02", eval_fraction=0.5, max_chars=9)
    prepare(workspace, path)
    assert split_contents("job02") == first


def test_explicit_eval_wins_duplicates_and_promotes_the_complete_connected_group(workspace):
    repo, source = workspace
    training, evaluation = source / "training.jsonl", source / "evaluation.jsonl"
    jsonl(training, [{"text": "Shared exact content", "family": "a"},
                     {"text": "Unique group content", "family": "a"}])
    jsonl(evaluation, [{"text": "Shared exact content"}])
    path = plan_file(workspace, [item(training, "i01", group_field="family"),
                               item(evaluation, "i02", split="eval")])
    prepare(workspace, path)
    assert not records(repo / "data/prepared/job01/text/en/train.jsonl")
    assert [row["text"] for row in records(repo / "data/prepared/job01/text/en/eval.jsonl")] == [
        "Shared exact content", "Unique group content"]
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    assert report["inputs"][0]["id"] == "i02"
    assert report["inputs"][1]["duplicates"] == 1


def test_later_hash_holdout_duplicates_cannot_leave_content_in_train(workspace):
    repo, source = workspace
    original = source / "duplicates.jsonl"
    first = item(original, namespace="connected", group_field="family")
    dummy = {"text": "Identical content", "lang": "en"}
    train_group = next(n for n in range(100) if not prepare_data._group({"family": n}, dummy, first, 7, 0.5)[1])
    eval_group = next(n for n in range(100) if prepare_data._group({"family": n}, dummy, first, 7, 0.5)[1])
    jsonl(original, [{"text": "Identical content", "family": train_group},
                     {"text": "Other connected content", "family": train_group},
                     {"text": "Identical content", "family": eval_group}])
    prepare(workspace, plan_file(workspace, [first], eval_fraction=0.5))
    assert not records(repo / "data/prepared/job01/text/en/train.jsonl")
    assert len(records(repo / "data/prepared/job01/text/en/eval.jsonl")) == 2


def test_malformed_json_lines_are_counted_privately(workspace):
    repo, source = workspace
    original = source / "malformed.jsonl"
    original.write_text('{"text":"Accepted"}\n{invalid\n[]\n{"text":""}\n{"missing":"field"}\n', encoding="utf-8")
    prepare(workspace, plan_file(workspace, [item(original)]))
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    stats = report["inputs"][0]
    assert (stats["rows"], stats["accepted"], stats["dropped"], stats["malformed"]) == (5, 1, 4, 2)
    assert records(repo / "data/prepared/job01/text/en/train.jsonl") == [{"text": "Accepted", "lang": "en"}]


def test_minimum_length_filters_documents_without_dropping_short_final_chunks(workspace):
    repo, source = workspace
    original = source / "lengths.jsonl"
    jsonl(original, [{"text": "short"}, {"text": "0123456789ABC"}])
    prepare(workspace, plan_file(workspace, [item(original)], min_chars=10, max_chars=10))
    assert [row["text"] for row in records(repo / "data/prepared/job01/text/en/train.jsonl")] == [
        "0123456789", "ABC"]


def test_completed_job_refuses_rerun_without_changing_files(workspace):
    repo, source = workspace
    original = source / "local.jsonl"
    jsonl(original, [{"text": "Complete job"}])
    path = plan_file(workspace, [item(original)])
    prepare(workspace, path)
    snapshot = {p: p.read_bytes() for p in (repo / "data").rglob("*") if p.is_file()}
    with pytest.raises(prepare_data.PreparationError, match="overwritten"):
        prepare(workspace, path)
    assert {p: p.read_bytes() for p in (repo / "data").rglob("*") if p.is_file()} == snapshot


@pytest.mark.parametrize("key,value", [("job", "../escape"), ("job", "/absolute"),
                                       ("id", "../escape"), ("lang", "../../escape")])
def test_unsafe_destination_names_are_rejected_before_import(workspace, key, value):
    repo, source = workspace
    original = source / "local.jsonl"
    jsonl(original, [{"text": "Input"}])
    input_item = item(original)
    options = {key: value} if key == "job" else {}
    if key != "job":
        input_item[key] = value
    with pytest.raises(prepare_data.PreparationError):
        prepare(workspace, plan_file(workspace, [input_item], **options))
    assert not (repo / "data").exists()


def test_plan_and_sources_must_stay_in_their_permitted_roots(workspace):
    repo, source = workspace
    original = source / "local.jsonl"
    jsonl(original, [{"text": "Input"}])
    path = plan_file(workspace, [item(original)])
    outside = repo / "public-plan.json"
    outside.write_bytes(path.read_bytes())
    with pytest.raises(prepare_data.PreparationError):
        prepare(workspace, outside)
    other = repo / "other.jsonl"
    jsonl(other, [{"text": "Other input"}])
    with pytest.raises(prepare_data.PreparationError):
        prepare(workspace, plan_file(workspace, [item(other)]))
    assert not (repo / "data").exists()


def test_checksum_failure_never_marks_a_job_complete(workspace):
    repo, source = workspace
    original = source / "local.jsonl"
    jsonl(original, [{"text": "Input"}])
    with pytest.raises(prepare_data.PreparationError, match="checksum"):
        prepare(workspace, plan_file(workspace, [item(original, sha256="0" * 64)]))
    assert not (repo / "data/prepared/job01/COMPLETE.json").exists()
    assert not (repo / "data/raw/input01.jsonl").exists()
    assert not list((repo / "data/raw").glob("*.part"))
    report = json.loads((repo / ".private/preparation-report-job01.json").read_text())
    assert report["status"] == "failed"


def test_source_changes_during_copy_are_detected(workspace, monkeypatch):
    repo, source = workspace
    original = source / "local.jsonl"
    jsonl(original, [{"text": "Input"}])
    path = plan_file(workspace, [item(original)])
    actual_progress = prepare_data.tqdm

    class MutatingProgress:
        def __init__(self, *args, **kwargs):
            self.bar = actual_progress(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.bar.close()

        def update(self, count):
            with original.open("ab") as stream:
                stream.write(b" ")

    monkeypatch.setattr(prepare_data, "tqdm", MutatingProgress)
    monkeypatch.setattr(prepare_data, "CHUNK_BYTES", original.stat().st_size + 1)
    # One mutation is sufficient; continuing to append would make the stream endless.
    calls = 0
    original_update = MutatingProgress.update

    def mutate_once(self, count):
        nonlocal calls
        if calls == 0:
            original_update(self, count)
        calls += 1

    monkeypatch.setattr(MutatingProgress, "update", mutate_once)
    with pytest.raises(prepare_data.PreparationError, match="changed"):
        prepare(workspace, path)
    assert not (repo / "data/prepared/job01/COMPLETE.json").exists()


def test_cli_outputs_only_counts_and_generic_errors(workspace, capsys):
    repo, source = workspace
    original = source / "private-synthetic-name.jsonl"
    jsonl(original, [{"text": "Private synthetic content"}])
    path = plan_file(workspace, [item(original)])
    assert prepare_data.main(["--repo", str(repo), "--plan", str(path), "--no-progress"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"counts": {"text": {"en": {"train": 1, "eval": 0}}}}
    assert not output.err
    assert prepare_data.main(["--repo", str(repo), "--plan", str(path), "--no-progress"]) == 1
    failure = capsys.readouterr()
    assert "private-synthetic-name" not in failure.out + failure.err
    assert "input01" not in output.out + failure.err
    assert "Private synthetic content" not in output.out + failure.err
