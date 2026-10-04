"""Read a calibration run's training log and decide the first-hour gate (ARCHITECTURE.md §11, D-017, D-018).

The paid run starts with a short calibration of each stage on the real data
and the real GPU. This tool turns that stage's train_log.jsonl into the
numbers the gate needs, without touching the trainer:

- seconds per optimizer step, overall and per task (from the cumulative
  "seconds" field, so batch building, logging and saves are included),
- valid (non-padding) and padded input/canvas positions per second, from
  each record's "batch_summary",
- CUDA allocated/reserved peaks against a memory limit,
- non-finite losses or gradient norms,
- whether each task's loss is falling (mean of the first vs the last window),
- and, for the hours allocated to the stage, the total_steps to use and
  the valid positions they project.

The log must be written with log_every=1 so that every step has a record.
The first --skip steps (CUDA warm-up, torch.compile) are left out of all
timing. The log rounds "seconds" to 0.1 s, so time at least a few hundred
steps for a per-step figure accurate to a few percent. Numbers are measurements of this run on this host; the projection
assumes the measured mix of tasks and depths continues.

Usage
  python tools/first_hour_gate.py runs/calib/base-dense/train_log.jsonl --hours 22 \
      --skip 20 --memory-limit-gib 76 --output .private/gate/base-dense.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

GIB = 1024 ** 3


def read_records(path: str | Path) -> list[dict]:
    """The log's records in step order; a resumed run's repeated steps keep the last copy."""
    by_step = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_step[record["step"]] = record
    return [by_step[step] for step in sorted(by_step)]


def step_times(records: list[dict]) -> list[tuple[dict, float]]:
    """(record, seconds spent on that step) for every step whose previous step is also logged.

    "seconds" counts from the start of the process, so a resumed run (where
    it restarts from zero) gives no time for its first logged step.
    """
    timed = []
    for previous, record in zip(records, records[1:]):
        elapsed = record["seconds"] - previous["seconds"]
        if record["step"] == previous["step"] + 1 and elapsed >= 0:
            timed.append((record, elapsed))
    return timed


def _finite(value) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(value))


def loss_trend(records: list[dict], window: int) -> dict:
    """Per task: mean loss of its first and last `window` records, and whether it fell."""
    trend = {}
    for task in sorted({r["task"] for r in records}):
        losses = [r["loss"] for r in records if r["task"] == task and _finite(r["loss"])]
        if len(losses) < 2 * window:
            trend[task] = {"records": len(losses), "falling": None}  # too few to tell
            continue
        first, last = statistics.fmean(losses[:window]), statistics.fmean(losses[-window:])
        trend[task] = {"records": len(losses), "first_mean": first, "last_mean": last,
                       "falling": last < first}
    return trend


def analyze(records: list[dict], hours: float, skip: int = 20, overhead: float = 0.9,
            memory_limit_gib: float | None = None, loss_window: int = 10) -> dict:
    if not records:
        raise ValueError("the log has no records")
    measured = [(r, t) for r, t in step_times(records) if r["step"] >= skip]
    if not measured:
        raise ValueError(f"no timed steps after --skip {skip}; log with log_every=1 and run longer")

    seconds = sum(t for _, t in measured)
    per_task = {}
    for task in sorted({r["task"] for r, _ in measured}):
        times = [t for r, t in measured if r["task"] == task]
        per_task[task] = {"steps": len(times), "mean_seconds_per_step": statistics.fmean(times)}

    sums = {key: 0 for key in ("valid_input_tokens", "padded_input_tokens", "valid_canvas_tokens",
                               "padded_canvas_tokens", "rows", "targets")}
    for record, _ in measured:
        summary = record.get("batch_summary") or {}
        for key in sums:
            sums[key] += summary.get(key, 0)
    valid = sums["valid_input_tokens"] + sums["valid_canvas_tokens"]
    padded = sums["padded_input_tokens"] + sums["padded_canvas_tokens"]

    mean_step = seconds / len(measured)
    steps = max(1, int(hours * 3600 * overhead / mean_step))
    valid_per_second = valid / seconds if seconds > 0 else 0.0

    peaks = [(r.get("cuda_peak_allocated_bytes"), r.get("cuda_peak_reserved_bytes")) for r in records]
    allocated = max((a for a, _ in peaks if a is not None), default=None)
    reserved = max((b for _, b in peaks if b is not None), default=None)
    nonfinite = [r["step"] for r in records if not (_finite(r.get("loss")) and _finite(r.get("grad_norm")))]
    trend = loss_trend(records, loss_window)

    checks = {
        "finite": not nonfinite,
        "loss_falling": all(t["falling"] is not False for t in trend.values())
                        and any(t["falling"] for t in trend.values()),
        "memory": (None if memory_limit_gib is None or reserved is None
                   else reserved / GIB <= memory_limit_gib),
    }
    return {
        "measured_steps": len(measured), "skipped_below_step": skip,
        "mean_seconds_per_step": mean_step, "per_task": per_task,
        "valid_positions_per_second": valid_per_second,
        "padded_positions_per_second": padded / seconds if seconds > 0 else 0.0,
        "valid_fraction": valid / padded if padded else None,
        "mean_rows_per_step": sums["rows"] / len(measured),
        "targets_per_second": sums["targets"] / seconds if seconds > 0 else 0.0,
        "cuda_peak_allocated_gib": None if allocated is None else allocated / GIB,
        "cuda_peak_reserved_gib": None if reserved is None else reserved / GIB,
        "nonfinite_steps": nonfinite,
        "loss_trend": trend,
        "plan": {"hours": hours, "overhead": overhead, "total_steps": steps,
                 "projected_valid_positions": valid_per_second * hours * 3600 * overhead},
        "checks": checks,
        "passed": all(value is not False for value in checks.values()),
    }


def summary_text(name: str, report: dict) -> str:
    checks = ", ".join(f"{key}={'n/a' if value is None else ('ok' if value else 'FAIL')}"
                       for key, value in report["checks"].items())
    plan = report["plan"]
    peak = report["cuda_peak_reserved_gib"]
    return "\n".join([
        f"{name}: {'PASS' if report['passed'] else 'FAIL'} ({checks})",
        f"  {report['measured_steps']} timed steps, {report['mean_seconds_per_step']:.3f} s/step, "
        f"{report['valid_positions_per_second']:,.0f} valid positions/s "
        f"(valid fraction {report['valid_fraction'] or 0:.3f}, {report['mean_rows_per_step']:.1f} rows/step)",
        f"  peak reserved: {'n/a' if peak is None else f'{peak:.1f} GiB'}",
        f"  {plan['hours']} h x {plan['overhead']} -> total_steps {plan['total_steps']:,}, "
        f"about {plan['projected_valid_positions'] / 1e9:.2f}B valid positions",
    ])


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="train_log.jsonl of a calibration run written with log_every=1")
    p.add_argument("--hours", type=float, required=True, help="hours allocated to this stage")
    p.add_argument("--skip", type=int, default=20, help="leave out steps below this one (warm-up)")
    p.add_argument("--overhead", type=float, default=0.9, help="usable fraction of the allocated time")
    p.add_argument("--memory-limit-gib", type=float, help="fail if the reserved CUDA peak exceeds this")
    p.add_argument("--loss-window", type=int, default=10, help="records averaged at each end of the loss trend")
    p.add_argument("--output", help="also write the report as JSON here")
    args = p.parse_args(argv)
    if not 0 < args.overhead <= 1 or args.hours <= 0:
        p.error("--hours must be positive and --overhead in (0, 1]")

    report = analyze(read_records(args.log), args.hours, args.skip, args.overhead,
                     args.memory_limit_gib, args.loss_window)
    print(summary_text(Path(args.log).parent.name or "stage", report))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        sys.exit(1)
    return report


if __name__ == "__main__":
    main()
