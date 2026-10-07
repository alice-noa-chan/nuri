"""Run a private training command with a deadline and bounded group shutdown.

This stops local work. A separate provider controller must release rented
resources, because stopping a process or VM does not end storage billing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_status(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def latest_metrics(folder: Path) -> tuple[dict, bool]:
    """Read numeric metrics only; ignore a last line still being written."""
    metrics, invalid, _ = _latest_metric_snapshot(folder)
    return metrics, invalid


def _latest_metric_snapshot(folder: Path) -> tuple[dict, bool, Path | None]:
    """Include the log identity so a new stage can start its own step counter."""
    files = list(folder.glob("*/train_log.jsonl"))
    direct = folder / "train_log.jsonl"
    if direct.is_file():
        files.append(direct)
    if not files:
        return {}, False, None
    path = max(files, key=lambda item: item.stat().st_mtime)
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 65536))
        lines = source.read().splitlines()
    for line in reversed(lines):
        try:
            raw = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        keys = ("step", "loss", "grad_norm", "tokens", "targets", "seconds", "lr")
        metrics = {
            key: raw[key] for key in keys if isinstance(raw.get(key), (int, float))
        }
        invalid = any(not math.isfinite(value) for value in metrics.values())
        return (
            {key: value for key, value in metrics.items() if math.isfinite(value)},
            invalid,
            path,
        )
    return {}, False, path


def signal_group(process, force: bool = False) -> None:
    """The child owns a dedicated process group; no unrelated PID is targeted."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.send_signal(signal.CTRL_BREAK_EVENT)
    except ProcessLookupError:
        pass


def group_is_alive(process) -> bool:
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def run_supervised(
    command: list[str], *, status: Path, output: Path, deadline: float, **options
) -> int:
    """Refuse concurrent writers; stale locks require an explicit ownership check."""
    owned = []
    try:
        for path in (status, output):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.with_name(path.name + ".owner")
            with lock.open("x", encoding="utf-8") as owner:
                owner.write(str(os.getpid()))
            owned.append(lock)
        return _run_supervised(
            command, status=status, output=output, deadline=deadline, **options
        )
    finally:
        for lock in owned:
            lock.unlink()


def _run_supervised(
    command: list[str],
    *,
    status: Path,
    output: Path,
    deadline: float,
    progress: Path | None = None,
    grace_seconds: float = 120,
    stall_seconds: float = 1800,
    poll_seconds: float = 5,
    clock=time.time,
    monotonic=time.monotonic,
    sleep=time.sleep,
    popen=subprocess.Popen,
    send=signal_group,
    group_alive=group_is_alive,
) -> int:
    values = (deadline, grace_seconds, stall_seconds, poll_seconds)
    if not command or not all(math.isfinite(value) for value in values):
        raise ValueError("A command and finite time limits are required")
    if min(grace_seconds, stall_seconds, poll_seconds) <= 0:
        raise ValueError("Grace, stall and polling intervals must be positive")
    started = clock()
    remaining = deadline - started
    if remaining <= 0:
        write_status(status, {"state": "refused", "reason": "deadline_already_passed"})
        return 75
    elapsed_start = monotonic()
    stop_requested = False

    def on_signal(signum, frame):
        # Signal handlers must not take locks or serialize checkpoints.
        nonlocal stop_requested
        stop_requested = True

    handlers = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("ab", buffering=0) as transcript:
        options = {"stdout": transcript, "stderr": subprocess.STDOUT}
        if os.name == "posix":
            options["start_new_session"] = True
        else:
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = popen(command, **options)
        try:
            for name in (signal.SIGTERM, signal.SIGINT):
                handlers[name] = signal.signal(name, on_signal)
            record = {
                "state": "running",
                "pid": process.pid,
                "started_at_epoch": started,
                "deadline_epoch": deadline,
            }
            last_size = output.stat().st_size
            last_progress = monotonic()
            last_metric_path = None
            last_metric_step = -1
            stop_at = None
            reason = None
            forced = False
            while process.poll() is None or group_alive(process):
                now = monotonic()
                size = output.stat().st_size
                if size != last_size:
                    last_progress, last_size = now, size
                metrics, invalid, metric_path = (
                    _latest_metric_snapshot(progress)
                    if progress is not None
                    else ({}, False, None)
                )
                step = metrics.get("step")
                if not invalid and type(step) is int and step >= 0:
                    if metric_path != last_metric_path or step > last_metric_step:
                        last_progress = now
                        last_metric_path, last_metric_step = metric_path, step
                if reason is None:
                    if process.poll() is not None:
                        reason = "orphaned_children"
                    elif stop_requested:
                        reason = "signal"
                    elif clock() >= deadline or now - elapsed_start >= remaining:
                        reason = "deadline"
                    elif invalid:
                        reason = "nonfinite_metrics"
                    elif now - last_progress >= stall_seconds:
                        reason = "no_progress"
                    if reason is not None:
                        stop_at = now
                        send(process)
                elif not forced and now - stop_at >= grace_seconds:
                    send(process, force=True)
                    forced = True
                elif forced and now - stop_at >= grace_seconds + 30:
                    record.update(
                        state="failed",
                        reason="group_did_not_exit",
                        forced=True,
                        checked_at_epoch=clock(),
                    )
                    write_status(status, record)
                    return 70
                record.update(
                    state="stopping" if reason else "running",
                    reason=reason,
                    checked_at_epoch=clock(),
                    metrics=metrics,
                    forced=forced,
                )
                write_status(status, record)
                sleep(poll_seconds)
            exit_code = process.wait()
            record.update(
                state="finished",
                reason=reason or "command_finished",
                exit_code=exit_code,
                ended_at_epoch=clock(),
                forced=forced,
            )
            write_status(status, record)
            if reason in ("nonfinite_metrics", "no_progress"):
                return 70
            return 75 if reason is not None else exit_code
        finally:
            if process.poll() is None or group_alive(process):
                send(process, force=True)
                process.wait(timeout=30)
            for name, previous in handlers.items():
                signal.signal(name, previous)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--deadline", required=True, type=float, help="absolute Unix UTC seconds"
    )
    parser.add_argument(
        "--progress", type=Path, help="private run folder containing stage logs"
    )
    parser.add_argument("--grace-seconds", type=float, default=120)
    parser.add_argument("--stall-seconds", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return run_supervised(
            command,
            status=args.status,
            output=args.output,
            deadline=args.deadline,
            progress=args.progress,
            grace_seconds=args.grace_seconds,
            stall_seconds=args.stall_seconds,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # Commands may contain private paths. Do not print a captured argv/traceback.
        print(
            "Supervisor could not finish safely; inspect the private status.",
            file=sys.stderr,
        )
        return 70


if __name__ == "__main__":
    sys.exit(main())
