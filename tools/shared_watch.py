"""Shared-file change watcher for multi-agent collaboration.

Each agent keeps its own snapshot in .watch/<agent>.json and sees changes
relative to that snapshot.

  watch --agent NAME   stream one event per change (for agents that can receive events)
  wait  --agent NAME   block until the next change, print it, exit (for agents that only run commands)
  check --agent NAME   print changes since this agent's last check, then update the snapshot
  ack   --agent NAME   update the snapshot without printing (run after your own edits)
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".watch"
PATTERNS = ["AGENTS.md", "CLAUDE.md", "docs/**/*.md", "configs/**/*.json"]
EXCLUDE_PREFIXES = ("docs/archive/",)
DIFF_LINES = 12
LINE_WIDTH = 160


def scan():
    files = {}
    for pattern in PATTERNS:
        for path in ROOT.glob(pattern):
            rel = path.relative_to(ROOT).as_posix()
            if path.is_file() and not rel.startswith(EXCLUDE_PREFIXES):
                text = path.read_bytes().decode("utf-8", errors="replace")
                files[rel] = text.replace("\r\n", "\n")
    return files


def snapshot_path(agent):
    return STATE_DIR / f"{agent}.json"


def load(agent):
    path = snapshot_path(agent)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save(agent, files):
    STATE_DIR.mkdir(exist_ok=True)
    tmp = snapshot_path(agent).with_suffix(".tmp")
    tmp.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
    # On Windows the rename fails while another process (e.g. a running
    # `watch`) has the snapshot open for reading, so retry briefly.
    for attempt in range(20):
        try:
            tmp.replace(snapshot_path(agent))
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def describe(old, new, max_lines):
    lines = []
    for rel in sorted(set(old) | set(new)):
        before, after = old.get(rel), new.get(rel)
        if before == after:
            continue
        if before is None:
            lines.append(f"[watch] ADDED {rel} ({after.count(chr(10)) + 1} lines)")
            continue
        if after is None:
            lines.append(f"[watch] DELETED {rel}")
            continue
        changed = [
            line
            for line in difflib.unified_diff(
                before.split("\n"), after.split("\n"), lineterm="", n=0
            )
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        added = sum(line.startswith("+") for line in changed)
        lines.append(f"[watch] CHANGED {rel} (+{added} -{len(changed) - added})")
        for line in changed[:max_lines]:
            lines.append(
                "    " + (line if len(line) <= LINE_WIDTH else line[:LINE_WIDTH] + "…")
            )
        if len(changed) > max_lines:
            lines.append(f"    … {len(changed) - max_lines} more changed lines")
    return lines


def emit(lines):
    for line in lines:
        print(line, flush=True)


def watch(agent, interval, settle, max_lines, once=False, timeout=None):
    if load(agent) is None:
        save(agent, scan())
    if not once:
        print(f"[watch] watching shared files as '{agent}'", flush=True)
    started = time.monotonic()
    pending_since = None
    while True:
        if timeout and pending_since is None and time.monotonic() - started >= timeout:
            print(f"[watch] no changes within {timeout:.0f}s", flush=True)
            return
        try:
            current = scan()
            # Re-read the snapshot every poll so a concurrent `ack` silences own edits.
            if current == load(agent):
                pending_since = None
            elif pending_since is None:
                pending_since = time.monotonic()
            elif time.monotonic() - pending_since >= settle:
                emit(describe(load(agent), current, max_lines))
                save(agent, current)
                if once:
                    return
                pending_since = None
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[watch] error: {exc}", file=sys.stderr, flush=True)
        time.sleep(interval)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["watch", "wait", "check", "ack"])
    parser.add_argument("--agent", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="seconds a change must stay stable before reporting",
    )
    parser.add_argument(
        "--max-lines", type=int, default=DIFF_LINES, help="diff lines shown per file"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="wait mode: give up after this many seconds",
    )
    args = parser.parse_args()

    if args.mode in ("watch", "wait"):
        once = args.mode == "wait"
        watch(
            args.agent,
            args.interval,
            args.settle,
            args.max_lines,
            once,
            args.timeout if once else None,
        )
        return

    current = scan()
    previous = load(args.agent)
    if args.mode == "check":
        if previous is None:
            print(
                f"[watch] first check for '{args.agent}': snapshot created, read the shared files directly"
            )
        else:
            emit(
                describe(previous, current, args.max_lines)
                or ["[watch] no changes since last check"]
            )
    save(args.agent, current)


if __name__ == "__main__":
    main()
