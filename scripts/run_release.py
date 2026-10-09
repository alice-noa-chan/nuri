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
2. evaluates the released checkpoint (scripts/evaluate.py) when the plan
   asks, reusing only a matching successful receipt and intact result;
3. exports a release folder (scripts/export_release.py) with the measured
   GPU-hours of that stage.

The released checkpoint is <stage>/final, or, for a stage with
"select_best": true, the weights with the best validation score that
training kept in <stage>/best-step<N>/ (named by <stage>/best.json).

Wall time per stage is stored in <out>/run_summary.json, so a re-run after a
crash keeps the hours already spent. Paths in the plan are used as given;
nothing here names a dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Plan keys passed straight through to scripts/train.py as --flags.
TRAIN_FLAGS = (
    "preset",
    "steps",
    "batch_size",
    "lr",
    "warmup",
    "save_every",
    "seed",
    "reasoning_examples",
    "grad_accum",
    "log_every",
    "tensorboard_dir",
    "cross_mode_weight",
    "save_every_minutes",
    "holdout_config",
    "valid_every",
    "stop_after_completed_steps",
    "recipe_config",
    "assessment_recipe_config",
    "patience",
)
# scripts/train.py exits with this code after saving on SIGTERM (nuri.trainer.INTERRUPTED_EXIT_CODE).
INTERRUPTED_EXIT_CODE = 75
# Plan keys that become bare switches when true (e.g. "compile": true -> --compile).
TRAIN_SWITCHES = (
    "gradient_checkpointing",
    "compile",
    "tensorboard",
    "no_progress",
    "pack_mlm",
    "bucket",
    "native_gqa",
    "liger_kernels",
    "select_best",
)
KERNEL_FLAGS = ("attention_backend", "vocab_loss_backend", "vocab_chunk_size")
DATA_FLAGS = {
    "base": ("corpus", "tokenizer"),
    "it": ("corpus", "tokenizer", "instructions"),
    "embed": ("tokenizer", "pairs"),
}


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def validate_holdout_settings(plan: dict) -> None:
    """Keep the optional holdout policy explicit for each model stage.

    The training CLI validates the private schema and input-manifest binding.
    Here we reject malformed plan values before any subprocess or output is
    created. Zero is a valid completion-only validation interval.
    """
    if any(
        key in plan.get("train", {})
        for key in ("holdout_config", "valid_every", "stop_after_completed_steps")
    ):
        raise ValueError(
            "holdout_config, valid_every and stop_after_completed_steps must be set per stage"
        )
    for name, spec in plan["stages"].items():
        if "holdout_config" in spec:
            config = spec["holdout_config"]
            if not isinstance(config, str) or not config.strip():
                raise ValueError(
                    f"{name}: holdout_config must be a nonempty path string"
                )
        if "valid_every" in spec:
            interval = spec["valid_every"]
            if type(interval) is not int or interval < 0:
                raise ValueError(f"{name}: valid_every must be an integer >= 0")
            if not spec.get("holdout_config"):
                raise ValueError(
                    f"{name}: valid_every requires an explicit holdout_config"
                )
        if spec.get("select_best") and not spec.get("holdout_config"):
            raise ValueError(f"{name}: select_best requires an explicit holdout_config")
        if "patience" in spec and (
            type(spec["patience"]) is not int
            or spec["patience"] < 0
            or (spec["patience"] and not spec.get("select_best"))
        ):
            raise ValueError(
                f"{name}: patience must be an integer >= 0 and needs select_best"
            )
        if "stop_after_completed_steps" in spec:
            boundary, steps = spec["stop_after_completed_steps"], spec.get("steps")
            if (
                type(boundary) is not int
                or type(steps) is not int
                or not 0 < boundary < steps
            ):
                raise ValueError(
                    f"{name}: stop_after_completed_steps must be an integer between 1 and steps - 1"
                )


def best_record(stage_dir: Path, spec: dict) -> dict | None:
    """The best-validation record of a select_best stage, or None."""
    path = stage_dir / "best.json"
    if not spec.get("select_best") or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def release_checkpoint(stage_dir: Path, spec: dict) -> Path:
    """The weights a stage evaluates and exports: final/ or its best validated step."""
    best = best_record(stage_dir, spec)
    if best is None:
        return stage_dir / "final"
    folder = stage_dir / best["folder"]
    if not (folder / "model.safetensors").is_file():
        raise ValueError(f"{stage_dir.name}: best.json names a missing checkpoint")
    return folder


def newest_state(stage_dir: Path) -> Path | None:
    states = sorted(
        (stage_dir / "training_state").glob("*.pt"), key=lambda p: p.stat().st_mtime
    )
    return states[-1] if states else None


def validate_recipe_settings(plan: dict) -> None:
    """Require a deliberate recipe path for each stage before launching work.

    Recipes describe one stage, so a plan-wide recipe would be ambiguous.
    The training CLI checks the actual private JSON and model geometry.
    """
    if any(
        key in plan.get("train", {})
        for key in ("recipe_config", "assessment_recipe_config")
    ):
        raise ValueError(
            "recipe_config and assessment_recipe_config must be set per stage"
        )
    for name, spec in plan["stages"].items():
        for key in ("recipe_config", "assessment_recipe_config"):
            if key not in spec:
                continue
            path = spec[key]
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{name}: {key} must be a nonempty path string")
            if key == "assessment_recipe_config" and not spec.get("holdout_config"):
                raise ValueError(
                    f"{name}: assessment_recipe_config requires holdout_config"
                )


def validate_kernel_settings(plan: dict) -> None:
    """Validate runtime kernel values without importing optional GPU packages."""
    for label, options in [("train", plan.get("train", {})), *plan["stages"].items()]:
        for key, choices in (
            ("attention_backend", ("sdpa", "flash2", "flash3")),
            ("vocab_loss_backend", ("torch", "chunked", "liger")),
        ):
            if key in options and options[key] not in choices:
                raise ValueError(f"{label}: {key} must be one of {choices}")
        if "vocab_chunk_size" in options and (
            type(options["vocab_chunk_size"]) is not int
            or options["vocab_chunk_size"] < 1
        ):
            raise ValueError(f"{label}: vocab_chunk_size must be a positive integer")
        if "liger_kernels" in options and type(options["liger_kernels"]) is not bool:
            raise ValueError(f"{label}: liger_kernels must be boolean")


def train_command(name: str, spec: dict, plan: dict, out: Path) -> list[str]:
    stage_dir = out / name
    command = [
        PYTHON,
        str(ROOT / "scripts" / "train.py"),
        "--stage",
        spec["stage"],
        "--out",
        str(stage_dir),
    ]
    if "init" in spec:  # the name of an earlier stage, or a checkpoint folder
        init = (
            out / spec["init"] / "final"
            if spec["init"] in plan["stages"]
            else Path(spec["init"])
        )
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
    for key in KERNEL_FLAGS:
        if key in spec or key in plan.get("train", {}):
            command += [flag(key), str(spec.get(key, plan.get("train", {}).get(key)))]
    for key in TRAIN_SWITCHES:
        if spec.get(key, plan.get("train", {}).get(key)):
            command.append(flag(key))
    state = newest_state(stage_dir)
    if state is not None:
        command += ["--resume", str(state)]
    return command


def evaluate_command(name: str, spec: dict, plan: dict, out: Path) -> list[str]:
    data = plan.get("data", {})
    command = [
        PYTHON,
        str(ROOT / "scripts" / "evaluate.py"),
        "--checkpoint",
        str(release_checkpoint(out / name, spec)),
        "--suites",
        *spec["evaluate"],
        "--output",
        str(out / "eval" / f"{name}.json"),
    ]
    if data.get("eval_corpus"):
        command += ["--corpus", str(data["eval_corpus"])]
    if data.get("eval_pairs"):
        command += ["--pairs", str(data["eval_pairs"])]
    if data.get("eval_instructions"):
        command += ["--instructions", str(data["eval_instructions"])]
    return command


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_source_sha256() -> str:
    """Bind the evaluator and model runtime sources, including relative names."""
    paths = [ROOT / "scripts" / "evaluate.py", *sorted((ROOT / "nuri").rglob("*.py"))]
    sources = [(str(path.relative_to(ROOT)), file_sha256(path)) for path in paths]
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def evaluation_input_hashes(command: list[str]) -> dict:
    """Stream only the resource bytes consumed by the requested suites."""
    start = command.index("--suites") + 1
    end = next(
        (i for i in range(start, len(command)) if command[i].startswith("--")),
        len(command),
    )
    suites = set(command[start:end])
    inputs = {}
    for option, environment, selected in (
        (
            "corpus",
            "NURI_EVAL_CORPUS",
            bool(suites & {"mlm", "denoise", "diagnostics"}),
        ),
        ("pairs", "NURI_EVAL_PAIRS", "retrieval" in suites),
        (
            "instructions",
            "NURI_EVAL_INSTRUCTIONS",
            bool(suites & {"instructions", "conditioning"}),
        ),
    ):
        if not selected:
            continue
        value = (
            command[command.index(flag(option)) + 1]
            if flag(option) in command
            else os.environ.get(environment)
        )
        if not value:
            raise ValueError(f"selected evaluation needs explicit {option} input")
        path = Path(value).resolve()
        paths = (
            [path / "tokens.bin", path / "offsets.npy"]
            if option == "corpus"
            else [path]
        )
        inputs[option] = [
            {"path": str(item), "sha256": file_sha256(item)} for item in paths
        ]
    return inputs


def evaluation_manifest_hashes(plan: dict, spec: dict) -> dict:
    """Bind explicitly supplied private manifests without hashing training data."""
    manifests = {}
    for location, block in (
        ("plan", plan),
        ("data", plan.get("data", {})),
        ("stage", spec),
    ):
        for key in ("input_manifest", "eval_input_manifest"):
            if block.get(key):
                path = Path(block[key]).resolve()
                manifests[f"{location}.{key}"] = {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
    # A stage holdout configuration may also identify the reviewed split/input
    # manifest. Its paths follow the training CLI's config-relative convention.
    if spec.get("holdout_config"):
        config_path = Path(spec["holdout_config"]).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("input_manifest"):
            path = Path(config["input_manifest"])
            if not path.is_absolute():
                path = config_path.parent / path
            manifests["holdout.input_manifest"] = {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
    return manifests


def evaluation_binding(
    command: list[str], stage_dir: Path, plan: dict, spec: dict
) -> dict:
    """Describe the actual released checkpoint and evaluator runtime for reuse."""
    checkpoint = release_checkpoint(stage_dir, spec)
    versions = {}
    for package in (
        "torch",
        "numpy",
        "tokenizers",
        "safetensors",
        "transformers",
        "scikit-learn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "config_sha256": file_sha256(checkpoint / "config.json"),
        "tokenizer_sha256": {
            name: file_sha256(checkpoint / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
            if (checkpoint / name).is_file()
        },
        "input_sha256": evaluation_input_hashes(command),
        "manifest_sha256": evaluation_manifest_hashes(plan, spec),
        "command": command,
        "runtime_source_sha256": evaluation_source_sha256(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "NURI_EVAL_CORPUS",
                "NURI_EVAL_PAIRS",
                "NURI_EVAL_INSTRUCTIONS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def _valid_evaluation_result(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, ValueError):
        return False


def _evaluation_matches(receipt: dict | None, binding: dict, output: Path) -> bool:
    return bool(
        isinstance(receipt, dict)
        and receipt.get("schema_version") == 1
        and receipt.get("returncode") == 0
        and receipt.get("binding") == binding
        and output.is_file()
        and receipt.get("output_sha256") == file_sha256(output)
        and _valid_evaluation_result(output)
    )


def _preserve_evaluation(output: Path, receipt: dict | None) -> None:
    """Copy prior evidence before replacement, including damaged untrusted output."""
    if not output.exists() and receipt is None:
        return
    history = output.parent / "history" / output.stem / str(time.time_ns())
    history.mkdir(parents=True)
    if output.is_file():
        shutil.copy2(output, history / output.name)
    if receipt is not None:
        (history / "receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )


def evaluate_stage(
    name: str,
    spec: dict,
    plan: dict,
    out: Path,
    record: dict,
    summary: dict,
    summary_path: Path,
    dry_run: bool,
) -> int:
    """Run or reuse a verified evaluation without discarding previous evidence."""
    command = evaluate_command(name, spec, plan, out)
    if dry_run:
        return run(command, True)
    output = out / "eval" / f"{name}.json"
    binding = evaluation_binding(command, out / name, plan, spec)
    receipt = record.get("evaluation_receipt")
    if _evaluation_matches(receipt, binding, output):
        print(f"[{name}] evaluation already verified, reusing result", flush=True)
        return 0
    _preserve_evaluation(output, receipt)
    if output.is_file():
        output.unlink()  # The preserved copy is evidence; a success must produce a fresh result.
    record.pop("evaluation_receipt", None)
    write_summary(summary_path, summary)
    code = run(command, False)
    if code != 0:
        return code
    if not _valid_evaluation_result(output):
        raise ValueError(f"{name}: evaluator did not produce a valid JSON object")
    # Reject source/checkpoint changes during assessment rather than issuing a
    # receipt for a result whose actual computation cannot be identified.
    if evaluation_binding(command, out / name, plan, spec) != binding:
        raise ValueError(f"{name}: evaluation binding changed while the evaluator ran")
    record["evaluation_receipt"] = {
        "schema_version": 1,
        "returncode": 0,
        "binding": binding,
        "output_sha256": file_sha256(output),
    }
    write_summary(summary_path, summary)
    return 0


def export_command(
    name: str, spec: dict, plan: dict, out: Path, hours: float
) -> list[str]:
    release = plan.get("release", {})
    command = [
        PYTHON,
        str(ROOT / "scripts" / "export_release.py"),
        "--checkpoint",
        str(release_checkpoint(out / name, spec)),
        "--out",
        str(out / "release" / spec["release_name"]),
        "--name",
        spec["release_name"],
        "--stage",
        spec["stage"],
        "--gpu-hours",
        f"{hours:.2f}",
    ]
    if release.get("gpu"):
        command += ["--gpu", release["gpu"]]
    best = best_record(out / name, spec)
    tokens = best["tokens"] if best else _trained_tokens(out / name)
    if tokens:
        command += ["--tokens", f"{tokens:.3g}"]
    return command


def _trained_tokens(stage_dir: Path) -> int | None:
    """Processed tokens from the last line of the stage's training log."""
    log = stage_dir / "train_log.jsonl"
    if not log.exists():
        return None
    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return json.loads(lines[-1]).get("tokens") if lines else None


class _ForwardSigterm:
    """While a child step runs, pass SIGTERM on to it and remember that it came.

    Without this, `kill <runner pid>` would stop the runner but leave
    scripts/train.py running without the chance to save. A host supervisor
    may still signal the whole process group; the child then simply sees
    SIGTERM once from each. (On Windows, send_signal(SIGTERM) terminates the
    child, so a clean save there needs a POSIX host.)
    """

    def __init__(self, child: subprocess.Popen):
        self.child = child
        self.received = False

    def _forward(self, signum, frame):
        # Lock-free on purpose (a second signal may re-enter this handler):
        # set a flag and signal the child's pid directly, without Popen.poll()
        # or send_signal(), which take Popen's internal lock.
        self.received = True
        try:
            os.kill(self.child.pid, signal.SIGTERM)
        except OSError:
            pass  # the child already exited

    def __enter__(self):
        self.previous = None
        if (
            hasattr(signal, "SIGTERM")
            and threading.current_thread() is threading.main_thread()
        ):
            self.previous = signal.signal(signal.SIGTERM, self._forward)
        return self

    def __exit__(self, *exc):
        if self.previous is not None:
            signal.signal(signal.SIGTERM, self.previous)
        return False


def run(command: list[str], dry_run: bool) -> int:
    """Run one step of the plan. Returns the exit code (0 on success or dry run).

    A stopped training run (exit code 75, state saved) is returned to the
    caller; any other failure raises as before, unless SIGTERM arrived while
    the step ran: then the plan stops with code 75 (rerun to continue).
    """
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    child = subprocess.Popen(command, cwd=ROOT)
    with _ForwardSigterm(child) as forward:
        code = child.wait()
    if forward.received:
        return INTERRUPTED_EXIT_CODE
    if code not in (0, INTERRUPTED_EXIT_CODE):
        raise subprocess.CalledProcessError(code, command)
    return code


def write_summary(path: Path, summary: dict) -> None:
    """Write run_summary.json atomically (temporary file, fsync, rename).

    A runner killed mid-write leaves the previous complete summary, so the
    next invocation can still read it and resume.
    """
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _finished(stage_dir: Path) -> bool:
    """A stage is done when its final weights and its final full state both exist."""
    return (stage_dir / "final" / "model.safetensors").exists() and (
        stage_dir / "training_state" / "final.pt"
    ).exists()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("plan")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="+", help="run only these stage names")
    args = p.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    validate_holdout_settings(plan)
    validate_recipe_settings(plan)
    validate_kernel_settings(plan)
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "run_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )

    for name, spec in plan["stages"].items():
        if args.only and name not in args.only:
            continue
        record = summary.setdefault(name, {"train_seconds": 0.0})
        if _finished(out / name):
            print(f"[{name}] already trained, skipping training", flush=True)
        else:
            started = time.time()
            try:
                code = run(train_command(name, spec, plan, out), args.dry_run)
            finally:
                # Count the attempt's time even when the step failed.
                if not args.dry_run:
                    record["train_seconds"] += time.time() - started
                    write_summary(summary_path, summary)
            if code == INTERRUPTED_EXIT_CODE:
                # Stop the whole plan; running the same command again resumes
                # this stage from its newest training state.
                print(
                    f"[{name}] training was stopped and saved; rerun this plan to resume",
                    flush=True,
                )
                sys.exit(INTERRUPTED_EXIT_CODE)
        if (
            spec.get("evaluate")
            and evaluate_stage(
                name, spec, plan, out, record, summary, summary_path, args.dry_run
            )
            == INTERRUPTED_EXIT_CODE
        ):
            print(
                f"[{name}] stopped during evaluation; rerun this plan to continue",
                flush=True,
            )
            sys.exit(INTERRUPTED_EXIT_CODE)
        best = None if args.dry_run else best_record(out / name, spec)
        if best is not None:
            record["selected_checkpoint"] = {
                key: best[key] for key in ("folder", "step", "score")
            }
            print(
                f"[{name}] releasing {best['folder']} "
                f"(best validation score {best['score']:.4f})",
                flush=True,
            )
            write_summary(summary_path, summary)
        if spec.get("release_name"):
            command = export_command(
                name, spec, plan, out, record["train_seconds"] / 3600
            )
            if run(command, args.dry_run) == INTERRUPTED_EXIT_CODE:
                print(
                    f"[{name}] stopped during export; rerun this plan to continue",
                    flush=True,
                )
                sys.exit(INTERRUPTED_EXIT_CODE)

    if not args.dry_run:
        write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
