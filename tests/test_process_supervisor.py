"""Exercise deadline and failure paths without a GPU or rented resources."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.supervise_process import latest_metrics, run_supervised, signal_group


class Clock:
    value = 100.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Child:
    pid = 12345

    def __init__(self, clock, finish=110):
        self.clock, self.finish = clock, finish

    def poll(self):
        return 0 if self.clock.now() >= self.finish else None

    def wait(self, timeout=None):
        return 0


def execute(tmp_path, *, deadline=105, finish=110, stall=100, grace=2):
    clock = Clock()
    child = Child(clock, finish)
    calls = []
    options = []

    def launch(command, **kwargs):
        options.append(kwargs)
        return child

    code = run_supervised(["fake"], status=tmp_path / "status.json", output=tmp_path / "stdout.log",
                          deadline=deadline, progress=tmp_path / "runs", grace_seconds=grace,
                          stall_seconds=stall, poll_seconds=1, clock=clock.now,
                          monotonic=clock.now, sleep=clock.sleep, popen=launch,
                          send=lambda proc, force=False: calls.append(force),
                          group_alive=lambda proc: proc.poll() is None)
    return code, json.loads((tmp_path / "status.json").read_text()), calls, options


def test_passed_deadline_never_launches(tmp_path):
    code, record, calls, options = execute(tmp_path, deadline=99)
    assert code == 75 and record["reason"] == "deadline_already_passed"
    assert calls == options == []


def test_deadline_terminates_then_escalates(tmp_path):
    code, record, calls, options = execute(tmp_path)
    assert code == 75 and record["reason"] == "deadline"
    assert calls[:2] == [False, True] and record["forced"]
    assert "start_new_session" in options[0] or "creationflags" in options[0]


def test_ordinary_completion_preserves_exit_code(tmp_path):
    code, record, calls, _ = execute(tmp_path, deadline=120, finish=102)
    assert code == 0 and record["reason"] == "command_finished"
    assert calls == []


def test_stall_stops_early(tmp_path):
    code, record, calls, _ = execute(tmp_path, deadline=120, stall=2)
    assert code == 70 and record["reason"] == "no_progress"
    assert calls[0] is False


def test_nonfinite_metric_is_removed_and_stops(tmp_path):
    folder = tmp_path / "runs" / "base"
    folder.mkdir(parents=True)
    (folder / "train_log.jsonl").write_text('{"step": 1, "loss": NaN, "tokens": 20}\n')
    code, record, calls, _ = execute(tmp_path, deadline=120)
    assert code == 70 and record["reason"] == "nonfinite_metrics"
    assert record["metrics"] == {"step": 1, "tokens": 20}
    assert calls[0] is False


def test_incomplete_log_line_keeps_last_complete_record(tmp_path):
    folder = tmp_path / "base"
    folder.mkdir()
    (folder / "train_log.jsonl").write_text('{"step": 2, "loss": 1.25}\n{"step":')
    assert latest_metrics(tmp_path) == ({"step": 2, "loss": 1.25}, False)


def test_missing_process_group_is_already_stopped(monkeypatch):
    import tools.supervise_process as module

    monkeypatch.setattr(module.os, "name", "posix")

    def absent(*args):
        raise ProcessLookupError

    monkeypatch.setattr(module.os, "killpg", absent, raising=False)
    signal_group(type("Gone", (), {"pid": 12345})())


def test_real_short_child_output_and_status(tmp_path):
    import sys
    import time

    code = run_supervised([sys.executable, "-c", "print('done', flush=True)"],
                          status=tmp_path / "status.json", output=tmp_path / "stdout.log",
                          deadline=time.time() + 20, poll_seconds=0.05)
    assert code == 0
    assert b"done" in (tmp_path / "stdout.log").read_bytes()
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "finished"


def test_existing_owner_prevents_a_second_launch(tmp_path):
    path = tmp_path / "status.json.owner"
    path.write_text("another owner")
    with pytest.raises(FileExistsError):
        execute(tmp_path)
    assert path.read_text() == "another owner"
    assert not (tmp_path / "stdout.log").exists()


def test_parent_exit_does_not_skip_group_grace(tmp_path):
    clock = Clock()
    child = Child(clock, finish=102)
    group = {"alive": True}
    calls = []

    def stop(process, force=False):
        calls.append((clock.now(), force))
        if force:
            group["alive"] = False

    code = run_supervised(["fake"], status=tmp_path / "status.json", output=tmp_path / "stdout.log",
                          deadline=120, grace_seconds=3, poll_seconds=1, clock=clock.now,
                          monotonic=clock.now, sleep=clock.sleep, popen=lambda *a, **k: child,
                          send=stop, group_alive=lambda process: group["alive"])
    assert code == 75
    assert calls == [(102, False), (105, True)]
    assert json.loads((tmp_path / "status.json").read_text())["reason"] == "orphaned_children"
