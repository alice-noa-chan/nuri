"""Exercise the local privacy guard using isolated repositories and fake markers."""

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_private_files.py"
FAKE_MARKER = "SYNTHETIC-PROVENANCE-TOKEN-43"
ZERO = "0" * 40


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("/data/\n/.private/\n/.dev/\n/.watch/\n", encoding="utf-8")
    git(tmp_path, "add", ".gitignore")
    return tmp_path


def commit(repo, message="Public change"):
    git(repo, "-c", "user.name=Local Test", "-c", "user.email=local@example.invalid",
        "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def write_public(repo, text, name="public.md"):
    (repo / name).write_text(text, encoding="utf-8")
    git(repo, "add", "--", name)


def write_markers(repo):
    (repo / ".private").mkdir(exist_ok=True)
    (repo / ".private/provenance-markers.txt").write_text(
        f"# Local literals, never disclosed by the guard\n\n{FAKE_MARKER}\n", encoding="utf-8")


def check(repo, *args, updates=""):
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), *args],
                          input=updates, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def push_update(local, remote=ZERO):
    return f"refs/heads/main {local} refs/heads/main {remote}\n"


@pytest.mark.parametrize("folder", ["data", ".private", ".dev", ".watch", "DATA"])
def test_forced_protected_additions_are_blocked_without_disclosing_paths(repo, folder):
    (repo / folder).mkdir()
    name = f"{folder}/private-synthetic-file.bin"
    (repo / name).write_bytes(b"local-only")
    git(repo, "add", "-f", "--", name)
    result = check(repo, "--pre-commit")
    assert result.returncode == 1
    assert "protected path" in result.stderr
    assert name not in result.stdout + result.stderr
    assert "private-synthetic-file" not in result.stdout + result.stderr


def test_generic_private_directory_documentation_is_allowed(repo):
    write_markers(repo)
    write_public(repo, "Keep local files in data/ and provenance in .private/. See .dev/ and .watch/.\n")
    assert check(repo).returncode == 0


def test_staged_markers_are_detected_even_if_the_working_copy_is_clean(repo):
    write_markers(repo)
    write_public(repo, FAKE_MARKER.lower())
    (repo / "public.md").write_text("Clean working copy", encoding="utf-8")
    result = check(repo)
    assert result.returncode == 1
    assert "content marker" in result.stderr
    assert FAKE_MARKER.casefold() not in (result.stdout + result.stderr).casefold()
    assert "public.md" not in result.stdout + result.stderr


def test_unstaged_markers_do_not_replace_the_staged_blob(repo):
    write_markers(repo)
    write_public(repo, "Clean staged copy")
    (repo / "public.md").write_text(FAKE_MARKER, encoding="utf-8")
    assert check(repo).returncode == 0


def test_deleting_a_previously_tracked_protected_file_is_allowed_at_commit(repo):
    (repo / "data").mkdir()
    (repo / "data/local.txt").write_text("local-only", encoding="utf-8")
    git(repo, "add", "-f", "data/local.txt")
    commit(repo)
    assert check(repo).returncode == 1  # Unchanged tracked private files count too.
    git(repo, "rm", "data/local.txt")
    assert check(repo).returncode == 0


def test_pre_push_catches_a_protected_file_removed_from_the_latest_tree(repo):
    baseline = commit(repo)
    (repo / "data").mkdir()
    (repo / "data/local.txt").write_text("local-only", encoding="utf-8")
    git(repo, "add", "-f", "data/local.txt")
    commit(repo)
    git(repo, "rm", "data/local.txt")
    tip = commit(repo)
    assert check(repo, "--audit-head").returncode == 0
    assert check(repo, "--pre-push", updates=push_update(tip, baseline)).returncode == 1


def test_pre_push_catches_a_marker_removed_from_public_history(repo):
    write_markers(repo)
    baseline = commit(repo)
    write_public(repo, FAKE_MARKER)
    commit(repo)
    write_public(repo, "Clean latest tree")
    tip = commit(repo)
    result = check(repo, "--pre-push", updates=push_update(tip, baseline))
    assert result.returncode == 1
    assert FAKE_MARKER not in result.stdout + result.stderr
    assert check(repo, "--audit-head").returncode == 0


def test_old_remote_history_does_not_block_a_clean_outgoing_range(repo):
    write_markers(repo)
    write_public(repo, FAKE_MARKER)
    remote = commit(repo)
    write_public(repo, "Clean outgoing tree")
    tip = commit(repo)
    assert check(repo, "--pre-push", updates=push_update(tip, remote)).returncode == 0
    assert check(repo, "--pre-push", updates=push_update(tip)).returncode == 1


def test_push_checks_commit_messages_and_annotated_tag_messages(repo):
    write_markers(repo)
    baseline = commit(repo)
    write_public(repo, "Clean tree")
    tip = commit(repo, FAKE_MARKER)
    assert check(repo, "--pre-push", updates=push_update(tip, baseline)).returncode == 1
    assert check(repo, "--audit-head").returncode == 1
    git(repo, "-c", "user.name=Local Test", "-c", "user.email=local@example.invalid",
        "tag", "-a", "local-tag", baseline, "-m", FAKE_MARKER)
    tag = git(repo, "rev-parse", "local-tag")
    result = check(repo, "--pre-push", updates=push_update(tag, baseline))
    assert result.returncode == 1
    assert FAKE_MARKER not in result.stdout + result.stderr


def test_commit_message_hook_scans_bytes_without_exposing_them(repo, tmp_path):
    write_markers(repo)
    message = tmp_path / "local-message.txt"
    message.write_text(FAKE_MARKER.lower(), encoding="utf-8")
    result = check(repo, "--commit-msg", str(message))
    assert result.returncode == 1
    assert FAKE_MARKER.casefold() not in (result.stdout + result.stderr).casefold()
    assert message.name not in result.stdout + result.stderr
    message.write_text("Explain generic data/ and .private/ handling", encoding="utf-8")
    assert check(repo, "--commit-msg", str(message)).returncode == 0


def test_clean_push_and_ref_deletion_are_allowed(repo):
    write_public(repo, "Public code and docs")
    tip = commit(repo)
    assert check(repo, "--pre-push", updates=push_update(tip)).returncode == 0
    assert check(repo, "--pre-push", updates=push_update(ZERO, tip)).returncode == 0


def test_missing_remote_object_checks_full_local_ancestry(repo):
    (repo / "data").mkdir()
    (repo / "data/local.txt").write_text("local-only", encoding="utf-8")
    git(repo, "add", "-f", "data/local.txt")
    commit(repo)
    git(repo, "rm", "data/local.txt")
    tip = commit(repo)
    assert check(repo, "--pre-push", updates=push_update(tip, "1" * 40)).returncode == 1


def test_check_failures_are_closed_and_do_not_print_private_paths(repo):
    result = check(repo, "--markers", ".private/missing-local-list.txt")
    assert result.returncode == 2
    assert "missing-local-list" not in result.stdout + result.stderr
    assert check(repo, "--pre-push", updates="malformed input\n").returncode == 2
