"""Run the whole release plan of ARCHITECTURE.md §11 with one command.

Usage
  python scripts/run_release.py plan.json            # run (or continue) every stage
  python scripts/run_release.py plan.json --dry-run  # only print the commands

The plan is a JSON file (keep real ones in .private/; see
configs/release_plan.example.json). For each stage, in order, the runner:

1. skips training if <out>/<stage>/final exists,
   resumes from the newest training_state/*.pt if the stage was interrupted,
   otherwise starts it;
   Plan-wide switches in "train" (for example {"tensorboard": true,
   "gradient_checkpointing": true}) apply to every stage unless the stage
   sets its own value.
2. evaluates the final checkpoint (scripts/evaluate.py) when the plan asks;
3. exports a release folder (scripts/export_release.py) with the measured
   GPU-hours of that stage.

Wall time per stage is stored in <out>/run_summary.json, so a re-run after a
crash keeps the hours already spent. Paths in the plan are used as given;
nothing here names a dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Plan keys passed straight through to scripts/train.py as --flags.
TRAIN_FLAGS = ("preset", "steps", "batch_size", "lr", "warmup", "save_every", "seed", "reasoning_examples",
               "grad_accum", "log_every", "tensorboard_dir", "cross_mode_weight", "save_every_minutes")
# scripts/train.py exits with this code after saving on SIGTERM (nuri.trainer.INTERRUPTED_EXIT_CODE).
INTERRUPTED_EXIT_CODE = 75
# Plan keys that become bare switches when true (e.g. "compile": true -> --compile).
TRAIN_SWITCHES = ("gradient_checkpointing", "compile", "tensorboard", "no_progress", "pack_mlm", "bucket", "native_gqa")
DATA_FLAGS = {"base": ("corpus", "tokenizer"), "it": ("corpus", "tokenizer", "instructions"),
              "embed": ("tokenizer", "pairs")}


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def newest_state(stage_dir: Path) -> Path | None:
    states = sorted((stage_dir / "training_state").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return states[-1] if states else None


def train_command(name: str, spec: dict, plan: dict, out: Path) -> list[str]:
    stage_dir = out / name
    command = [PYTHON, str(ROOT / "scripts" / "train.py"), "--stage", spec["stage"], "--out", str(stage_dir)]
    if "init" in spec:  # the name of an earlier stage, or a checkpoint folder
        init = out / spec["init"] / "final" if spec["init"] in plan["stages"] else Path(spec["init"])
        command += ["--init", str(init)]
    for key in DATA_FLAGS[spec["stage"]]:
        value = spec.get(key, plan.get("data", {}).get(key))
        if isinstance(value, list):  # e.g. several weighted corpora: ["ko=35", "en=45"]
            command += [flag(key), *map(str, value)]
        elif value is not None:
            command += [flag(key), str(value)]
    for key in TRAIN_FLAGS:
        if key in spec:
            command += [flag(key), str(spec[key])]
    for key in TRAIN_SWITCHES:
        if spec.get(key, plan.get("train", {}).get(key)):
            command.append(flag(key))
    state = newest_state(stage_dir)
    if state is not None:
        command += ["--resume", str(state)]
    return command


def evaluate_command(name: str, spec: dict, plan: dict, out: Path) -> list[str]:
    data = plan.get("data", {})
    command = [PYTHON, str(ROOT / "scripts" / "evaluate.py"), "--checkpoint", str(out / name / "final"),
               "--suites", *spec["evaluate"], "--output", str(out / "eval" / f"{name}.json")]
    if data.get("eval_corpus"):
        command += ["--corpus", str(data["eval_corpus"])]
    if data.get("eval_pairs"):
        command += ["--pairs", str(data["eval_pairs"])]
    if data.get("eval_instructions"):
        command += ["--instructions", str(data["eval_instructions"])]
    return command


def export_command(name: str, spec: dict, plan: dict, out: Path, hours: float) -> list[str]:
    release = plan.get("release", {})
    command = [PYTHON, str(ROOT / "scripts" / "export_release.py"), "--checkpoint", str(out / name / "final"),
               "--out", str(out / "release" / spec["release_name"]), "--name", spec["release_name"],
               "--stage", spec["stage"], "--gpu-hours", f"{hours:.2f}"]
    if release.get("gpu"):
        command += ["--gpu", release["gpu"]]
    tokens = _trained_tokens(out / name)
    if tokens:
        command += ["--tokens", f"{tokens:.3g}"]
    return command


def _trained_tokens(stage_dir: Path) -> int | None:
    """Processed tokens from the last line of the stage's training log."""
    log = stage_dir / "train_log.jsonl"
    if not log.exists():
        return None
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]).get("tokens") if lines else None


def run(command: list[str], dry_run: bool) -> int:
    """Run one step of the plan. Returns the exit code (0 on success or dry run).

    A stopped training run (exit code 75, state saved) is returned to the
    caller; any other failure raises as before.
    """
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    code = subprocess.run(command, cwd=ROOT).returncode
    if code not in (0, INTERRUPTED_EXIT_CODE):
        raise subprocess.CalledProcessError(code, command)
    return code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="+", help="run only these stage names")
    args = p.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    for name, spec in plan["stages"].items():
        if args.only and name not in args.only:
            continue
        record = summary.setdefault(name, {"train_seconds": 0.0})
        if (out / name / "final" / "model.safetensors").exists():
            print(f"[{name}] already trained, skipping training", flush=True)
        else:
            started = time.time()
            code = run(train_command(name, spec, plan, out), args.dry_run)
            if not args.dry_run:
                record["train_seconds"] += time.time() - started
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if code == INTERRUPTED_EXIT_CODE:
                # Stop the whole plan; running the same command again resumes
                # this stage from its newest training state.
                print(f"[{name}] training was stopped and saved; rerun this plan to resume", flush=True)
                sys.exit(INTERRUPTED_EXIT_CODE)
        if spec.get("evaluate"):
            run(evaluate_command(name, spec, plan, out), args.dry_run)
        if spec.get("release_name"):
            run(export_command(name, spec, plan, out, record["train_seconds"] / 3600), args.dry_run)

    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
