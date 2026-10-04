import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import publish_release as pub  # noqa: E402


def release_folder(folder: Path, ok=True, extra=None, readme="# card\n"):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "LICENSE", "tokenizer.json", "model.py"):
        (folder / name).write_text("{}" if name.endswith(".json") else "x", encoding="utf-8")
    (folder / "README.md").write_text(readme, encoding="utf-8")
    if extra:
        (folder / extra).write_text("x", encoding="utf-8")
    (folder.parent / f"{folder.name}.verify.json").write_text(json.dumps({"ok": ok}), encoding="utf-8")


@pytest.fixture
def run(tmp_path, monkeypatch):
    out = tmp_path / "out"
    plan = {"out": str(out), "stages": {
        "base-dense": {"stage": "base", "release_name": "nuri-base"},
        "it-dense": {"stage": "it", "release_name": "nuri-base-it"},
        "calibration": {"stage": "base"}}}  # no release_name: never published
    release_folder(out / "release" / "nuri-base")
    release_folder(out / "release" / "nuri-base-it")
    for name in ("step-2000", "step-10000", "pre-decay"):
        (out / "base-dense" / name).mkdir(parents=True)
        (out / "base-dense" / name / "model.safetensors").write_text("x", encoding="utf-8")
    monkeypatch.setattr(pub, "export_intermediate",
                        lambda checkpoint, folder, name, stage: release_folder(folder))
    monkeypatch.setattr(pub, "MARKERS", tmp_path / "markers.txt")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, out, tmp_path


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, private, exist_ok):
        self.calls.append(("repo", repo_id, private))

    def create_branch(self, repo_id, branch, exist_ok):
        self.calls.append(("branch", repo_id, branch))

    def upload_folder(self, repo_id, folder_path, revision, commit_message):
        self.calls.append(("upload", repo_id, revision))


def test_dry_run_lists_main_and_revisions_in_order_and_uploads_nothing(run):
    path, _, _ = run
    api = FakeApi()
    uploads = pub.main([str(path), "--intermediate"], api=api)
    assert [(u["repo"], u["revision"]) for u in uploads] == [
        ("nuri-base", "main"), ("nuri-base", "step-2000"), ("nuri-base", "step-10000"),
        ("nuri-base", "pre-decay"), ("nuri-base-it", "main")]
    assert api.calls == []


def test_push_needs_namespace_and_visibility(run):
    path, _, _ = run
    with pytest.raises(SystemExit):
        pub.main([str(path), "--push"], api=FakeApi())


def test_push_creates_repos_branches_and_uploads(run):
    path, _, _ = run
    api = FakeApi()
    pub.main([str(path), "--intermediate", "--push", "--namespace", "me", "--visibility", "private"], api=api)
    assert ("repo", "me/nuri-base", True) in api.calls
    assert ("branch", "me/nuri-base", "pre-decay") in api.calls
    assert ("upload", "me/nuri-base", "step-10000") in api.calls
    assert ("upload", "me/nuri-base-it", "main") in api.calls
    assert not any(c[0] == "branch" and c[2] == "main" for c in api.calls)


@pytest.mark.parametrize("problem", ["training_state.pt", "unverified", "marker"])
def test_any_failed_check_blocks_every_upload(run, problem):
    path, out, tmp_path = run
    folder = out / "release" / "nuri-base-it"
    if problem == "training_state.pt":
        release_folder(folder, extra="training_state.pt")
    elif problem == "unverified":
        release_folder(folder, ok=False)
    else:
        (tmp_path / "markers.txt").write_text("# private\nSecret-Source\n", encoding="utf-8")
        release_folder(folder, readme="trained on secret-source text\n")
    api = FakeApi()
    with pytest.raises(SystemExit):
        pub.main([str(path), "--push", "--namespace", "me", "--visibility", "public"], api=api)
    assert api.calls == []  # one bad folder stops the whole publication


def test_marker_matches_are_not_printed(run, capsys):
    path, out, tmp_path = run
    (tmp_path / "markers.txt").write_text("Secret-Source\n", encoding="utf-8")
    release_folder(out / "release" / "nuri-base", readme="secret-source\n")
    with pytest.raises(SystemExit):
        pub.main([str(path)], api=FakeApi())
    captured = capsys.readouterr()
    assert "secret" not in (captured.out + captured.err).casefold()


def test_tokenizer_repository_is_built_checked_and_listed_first(run, monkeypatch):
    path, out, _ = run
    monkeypatch.setattr(pub, "verify_tokenizer", lambda folder, release: {
        "vocab_size": 128, "pad_id": 0, "mask_id": 1, "eos_id": 2, "round_trip": True, "ok": True})
    api = FakeApi()
    uploads = pub.main([str(path), "--tokenizer-repo", "nuri-tokenizer", "--push", "--namespace", "me",
                        "--visibility", "public"], api=api)
    assert (uploads[0]["repo"], uploads[0]["kind"]) == ("nuri-tokenizer", "tokenizer")
    folder = out / "release" / "nuri-tokenizer"
    assert sorted(p.name for p in folder.iterdir()) == ["LICENSE", "README.md", "tokenizer.json"]
    assert "model.safetensors" not in {p.name for p in folder.iterdir()}
    assert ("upload", "me/nuri-tokenizer", "main") in api.calls and ("repo", "me/nuri-tokenizer", False) in api.calls


def test_releases_with_different_tokenizers_block_publication(run, monkeypatch):
    path, out, _ = run
    (out / "release" / "nuri-base-it" / "tokenizer.json").write_text('{"different": true}', encoding="utf-8")
    api = FakeApi()
    with pytest.raises(SystemExit):
        pub.main([str(path), "--tokenizer-repo", "nuri-tokenizer", "--push", "--namespace", "me",
                  "--visibility", "public"], api=api)
    assert api.calls == []
