"""Turn measured GPU benchmarks into step counts for a release plan (ARCHITECTURE.md §3, §11, §15).

Usage
  python scripts/plan_budget.py plan.json --out plan.budgeted.json

The plan gets a "budget" block naming the hours of each stage and the
tools/benchmark_nuri.py results (with --training) for each kind of batch the
stage trains on:

  "budget": {
    "overhead": 0.9,
    "stages": {
      "base-dense": {"hours": 8, "benchmarks": {"mlm": "bench/dense-mlm.json",
                                                 "span": "bench/dense-denoise.json"}},
      "it-dense":   {"hours": 1.5, "benchmarks": {"instruction": "bench/dense-denoise.json"}}
    }
  }

For each budgeted stage the script
- averages the measured seconds per step over the stage's task mix
  (base: mlm 0.5 / span 0.5; it: instruction + reasoning + span, all canvas
  batches; embed: pairs), times grad_accum,
- sets "steps" so that hours * overhead are used, and "save_every" so a
  checkpoint lands about every 1B processed tokens (§15),
- reports processed tokens. For the dense base stage it applies the size
  rule of §3: fewer than 3B tokens in the budget means the release preset
  is too large for this GPU, and the report says so.

Benchmarks must use the same batch size and lengths as the plan's stage;
a mismatch is reported, because seconds per step only transfer between
identical shapes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Task mix per stage kind (nuri/pipeline.py). Each benchmark key may cover several tasks.
STAGE_MIX = {
    "base": {"mlm": 0.5, "span": 0.5},
    "it": {"instruction": 0.7, "reasoning": 0.15, "span": 0.15},
    "embed": {"pairs": 1.0},
}
# Tasks without their own benchmark use this one: all IT batches are canvas batches.
SAME_SHAPE = {"reasoning": "instruction", "span": "instruction"}
SIZE_RULE_TOKENS = 3e9
CHECKPOINT_TOKENS = 1e9


def load_benchmark(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("training"):
        raise ValueError(f"{path}: benchmark was not run with --training")
    return data


def tokens_per_iteration(bench: dict) -> float:
    """Observed input tokens (both sides for pairs) plus canvas positions."""
    return bench["batch_size"] * (bench["input_length"] * bench.get("input_sequences_per_sample", 1)
                                 + (bench.get("output_length") or 0))


def budget_stage(name: str, stage: dict, budget: dict, overhead: float) -> dict:
    benchmarks = {kind: load_benchmark(path) for kind, path in budget["benchmarks"].items()}
    mix = STAGE_MIX[stage["stage"]]
    seconds, tokens, warnings = 0.0, 0.0, []
    for task, weight in mix.items():
        key = task if task in benchmarks else SAME_SHAPE.get(task)
        if key not in benchmarks:
            raise ValueError(f"{name}: no benchmark for task {task!r}")
        bench = benchmarks[key]
        seconds += weight * bench["latency_seconds"]["mean"]
        tokens += weight * tokens_per_iteration(bench)
        if "batch_size" in stage and bench["batch_size"] != stage["batch_size"]:
            warnings.append(f"{key} benchmark batch {bench['batch_size']} != plan batch {stage['batch_size']}")
    accumulation = stage.get("grad_accum", 1)
    seconds_per_step, tokens_per_step = seconds * accumulation, tokens * accumulation
    steps = max(1, int(budget["hours"] * 3600 * overhead / seconds_per_step))
    report = {"stage": name, "hours": budget["hours"], "seconds_per_step": round(seconds_per_step, 4),
              "steps": steps, "tokens_per_step": tokens_per_step, "processed_tokens": steps * tokens_per_step,
              "save_every": max(1, math.ceil(CHECKPOINT_TOKENS / tokens_per_step)), "warnings": warnings}
    if stage["stage"] == "base" and stage.get("preset", "release-dense") == "release-dense":
        report["size_rule"] = ("ok" if report["processed_tokens"] >= SIZE_RULE_TOKENS else
                               "shrink: fewer than 3B tokens in the budget; use D=512, L=8 (ARCHITECTURE.md §3)")
    return report


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--out", help="write the plan with filled-in steps and save_every here")
    args = p.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    budget = plan.get("budget", {})
    overhead = budget.get("overhead", 0.9)
    reports = []
    for name, stage_budget in budget.get("stages", {}).items():
        if name not in plan["stages"]:
            raise ValueError(f"budget names an unknown stage: {name}")
        report = budget_stage(name, plan["stages"][name], stage_budget, overhead)
        plan["stages"][name]["steps"] = report["steps"]
        if plan["stages"][name]["stage"] == "base":
            plan["stages"][name]["save_every"] = report["save_every"]
        reports.append(report)

    for r in reports:
        line = (f"{r['stage']:<12} {r['hours']:>5.2f} h  {r['seconds_per_step']:.3f} s/step  {r['steps']:>8,} steps  "
                f"{r['processed_tokens'] / 1e9:6.2f}B tokens")
        print(line + (f"  [{r['size_rule']}]" if "size_rule" in r else ""))
        for warning in r["warnings"]:
            print(f"  warning: {warning}")
    if args.out:
        Path(args.out).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return reports


if __name__ == "__main__":
    main()
