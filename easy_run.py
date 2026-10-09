"""One-command release training on a single A100 80GB host without internet.

Usage (on the GPU host, inside the unpacked bundle folder):
  python3 easy_run.py

Run the same command again after any interruption; it continues where it
stopped. Everything is driven by easy_run.json next to this file (made by
tools/pack_easy_run.py from a private copy of configs/easy_run.example.json).
No other arguments are needed and nothing is downloaded: the bundle carries
its own Python, the pinned packages and the data.

What it does, in order. Each finished step is recorded in
easy_run_work/state.json and skipped on the next run:

1. preflight   Check the launcher Python, the NVIDIA driver and GPU, free
               disk space and the bundled offline files.
2. install     Unpack the bundled CPython 3.11 into easy_run_work/python,
               create easy_run_work/venv and install the pinned wheels from
               offline/wheels with pip --no-index. Check that PyTorch sees
               CUDA.
3. data        Check every data file against data/SHA256SUMS (once).
4. calibrate   Train each release stage for a few steps on the real data,
               pick a micro-batch that fits in GPU memory (the effective
               batch stays fixed; accumulation makes up the rest) and
               measure seconds per optimizer step.
5. plan        Turn the time budget into steps per stage and write the
               release plan (frozen once written).
6. train       Run scripts/run_release.py: train, evaluate and export every
               stage. It resumes interrupted stages from their newest state.
               Failures are retried a few times; out of memory turns on
               gradient checkpointing for the unfinished stages first.
7. package     Copy models, evaluations, logs and summaries into
               easy_run_output/ and pack easy_run_output.tar for retrieval.

The budget counts wall-clock hours from the first start (stored in
state.json), so a restart does not grant extra time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "easy_run.json"
WORK = ROOT / "easy_run_work"
STATE_PATH = WORK / "state.json"
LOG_PATH = WORK / "easy_run.log"
OUTPUT = ROOT / "easy_run_output"
ARCHIVE = ROOT / "easy_run_output.tar"
OFFLINE = ROOT / "offline"  # bundled CPython archive, wheels and requirements.txt
VENV = WORK / "venv"
DATA_MARKER = WORK / "data_verified"  # holds the SHA256 of the checked data list
PYTHON = sys.executable  # replaced by the venv Python once it is installed
INTERRUPTED_EXIT_CODE = 75  # scripts/train.py and run_release.py after a saved stop

# The release stages (ARCHITECTURE.md D-004, D-005, D-007), in training order.
# "init" names the stage whose final checkpoint starts this one. The three
# dense products come first: if the host stops the job early, only the MoE
# variants are missing, never the main model family.
STAGES = {
    "base-dense": {
        "stage": "base",
        "preset": "release-dense",
        "release_name": "nuri-base",
    },
    "it-dense": {"stage": "it", "init": "base-dense", "release_name": "nuri-base-it"},
    "embed": {
        "stage": "embed",
        "init": "base-dense",
        "release_name": "nuri-base-embed",
    },
    "base-moe": {
        "stage": "base",
        "preset": "release-moe",
        "release_name": "nuri-base-moe",
    },
    "it-moe": {"stage": "it", "init": "base-moe", "release_name": "nuri-base-moe-it"},
}
EVALUATE = {
    "base": ["mlm", "denoise", "diagnostics"],
    "it": ["reasoning", "curves", "interventions"],
    "embed": ["retrieval"],
}
# Measured steps per stage; the first third is warmup. 30 steps logged every
# step overestimated the A100 step time by about 21% (0.647 s vs 0.513 s in
# the real run of base-dense), which would leave about two days of the window
# unused; 80 steps logged like training measured 0.541 s.
CALIBRATION_STEPS = 100
CALIBRATION_LOG_EVERY = 10  # the training default, so logging costs the same
MAX_MICRO_BATCH = 64  # the fastest measured A100 batch (docs/GPU_CAPACITY.md)
MIN_MICRO_BATCH = 4
MEMORY_HEADROOM = 0.85  # calibration peaks above this share of GPU memory are too tight
TRAIN_ATTEMPTS = 4  # run_release restarts after failures (each resumes)
OOM_MARKERS = ("out of memory", "outofmemoryerror")
MIN_BASE_STEPS = 100  # a base stage shorter than this is not worth running
MIN_FREE_DISK_GB = 300
PATIENCE = 5  # validations without a new best (a quarter of an IT/embed stage)
STOP_GRACE_SECONDS = 20 * 60  # a stopped child that has not exited by then is killed


# ----- small helpers -------------------------------------------------------------------


def log(message: str) -> None:
    """Print with a UTC timestamp and append to easy_run_work/easy_run.log."""
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {message}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def now() -> float:
    """Wall-clock seconds; the budget clock starts at the first run."""
    return time.time()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"done": [], "started_at": now()}


def save_state(state: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_PATH)


def run(command: list[str], **kwargs) -> int:
    """Run a command in the project folder, streaming its output; returns the exit code."""
    log("$ " + " ".join(command))
    return subprocess.run(command, cwd=ROOT, **kwargs).returncode


# The child started by run_logged, so a SIGTERM to easy_run can be passed on.
_child: subprocess.Popen | None = None


def request_stop(process: subprocess.Popen, reason: str) -> None:
    """Ask a training child to save and exit (scripts/train.py handles SIGTERM)."""
    log(f"stopping the running step: {reason}")
    try:
        os.kill(process.pid, signal.SIGTERM)
    except OSError:
        pass  # it has already exited


def on_sigterm(signum, frame) -> None:
    """easy_run itself was asked to stop: let the child save first, then package.

    With a child running, the child gets SIGTERM, saves its state and exits
    with INTERRUPTED_EXIT_CODE; the caller then stops and main() packages
    what is finished. Without a child, easy_run stops right away.
    """
    if _child is not None:
        request_stop(_child, "easy_run received SIGTERM")
    else:
        raise SystemExit("easy_run received SIGTERM; run it again to continue")


def run_logged(
    command: list[str], name: str, stop_at: float | None = None
) -> tuple[int, str]:
    """Run a command, streaming its output to the console and easy_run_work/logs/<name>.log.

    Returns the exit code and the last lines of output, so the caller can tell
    an out-of-memory failure from other failures. With stop_at (a now()
    time), the command is asked to stop at that moment and killed if it is
    still running STOP_GRACE_SECONDS later.
    """
    global _child
    log("$ " + " ".join(command))
    path = WORK / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=400)
    with path.open("a", encoding="utf-8") as handle, subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:  # fmt: skip
        timers = []
        if stop_at is not None:
            delay = max(0.0, stop_at - now())
            timers = [
                threading.Timer(
                    delay, request_stop, (process, "the time budget is used up")
                ),
                threading.Timer(delay + STOP_GRACE_SECONDS, process.kill),
            ]
        _child = process
        for timer in timers:
            timer.daemon = True
            timer.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                tail.append(line)
        finally:
            _child = None
            for timer in timers:
                timer.cancel()
    return process.returncode, "".join(tail)


def is_out_of_memory(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def check(command: list[str]) -> None:
    code = run(command)
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}: {command[:3]}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ----- configuration -------------------------------------------------------------------


def load_config(path: Path | None = None) -> dict:
    """Read easy_run.json and check every field easy_run needs."""
    path = path or CONFIG_PATH
    if not path.exists():
        raise SystemExit(
            f"{path.name} is missing. Build the bundle with tools/pack_easy_run.py, "
            "which puts the filled-in config next to this file."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise SystemExit("easy_run.json needs schema_version 1")
    inputs = config.get("inputs", {})
    for key in ("tokenizer", "corpus", "instructions", "pairs"):
        if not inputs.get(key):
            raise SystemExit(f"inputs.{key} is required")
    for split in ("valid", "test"):
        if not config.get("holdouts", {}).get(split, {}).get("corpus"):
            raise SystemExit(f"holdouts.{split}.corpus is required")
    budget = config.get("budget", {})
    shares = budget.get("shares", {})
    if set(shares) != set(STAGES) or any(v <= 0 for v in shares.values()):
        raise SystemExit(
            f"budget.shares needs a positive share for each of {list(STAGES)}"
        )
    if not 0 < budget.get("total_hours", 0) <= 24 * 30:
        raise SystemExit("budget.total_hours must be between 0 and 720")
    if not 0 <= budget.get("package_hours", 1.0) < budget["total_hours"]:
        raise SystemExit(
            "budget.package_hours must be at least 0 and below total_hours"
        )
    if not 0 < budget.get("safety", 0.95) <= 1:
        raise SystemExit("budget.safety must be in (0, 1]")
    train = config.get("train", {})
    for key, default in (("batch_size", 64), ("embed_batch_size", 128)):
        value = train.get(key, default)
        if type(value) is not int or value < 1 or value & (value - 1):
            raise SystemExit(
                f"train.{key} must be a power of two (it is split into micro-batches)"
            )
    return config


def data_dir(config: dict) -> Path:
    """The data folder of the bundle (absolute paths are used as given)."""
    return ROOT / config.get("data", {}).get("folder", "data")


def data_manifest(config: dict) -> Path:
    return data_dir(config) / config.get("data", {}).get("manifest", "SHA256SUMS")


def use_python(state: dict) -> None:
    """Run every later command with the installed venv Python, once there is one."""
    global PYTHON
    PYTHON = state.get("python", PYTHON)


def resolve(config: dict, value):
    """Make input paths (optionally "path=weight") absolute inside the unpacked data."""
    if isinstance(value, list):
        return [resolve(config, item) for item in value]
    path = strip_weight(value)
    weight = value[len(path) :]  # "" or "=0.6"
    return str(data_dir(config) / path) + weight


# ----- steps ---------------------------------------------------------------------------


def preflight(config: dict, state: dict) -> None:
    """Fail early on a host that cannot run the plan; record what was found."""
    if sys.version_info < (3, 8):
        raise SystemExit("python3 3.8 or newer is needed to start easy_run.py")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        log(
            f"warning: only {free_gb:.0f} GB free; {MIN_FREE_DISK_GB} GB or more is recommended"
        )
    offline_files(check_only=True)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"nvidia-smi failed ({error}); an NVIDIA driver is needed")
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(f"nvidia-smi found no GPU: {result.stderr.strip()}")
    name, memory_mib, driver = [
        part.strip() for part in result.stdout.strip().splitlines()[0].split(",")
    ]
    host = {
        "gpu": name,
        "memory_gb": float(memory_mib) * 1024**2 / 1e9,
        "driver": driver,
        "free_disk_gb": round(free_gb),
    }
    if "A100" not in name or host["memory_gb"] < 79:
        log(
            f"warning: expected an A100 80GB, found {name} ({host['memory_gb']:.0f} GB)"
        )
    state["host"] = host
    log(f"host: {host}")


def offline_files(check_only: bool = False) -> tuple[Path, Path, Path]:
    """The bundled CPython archive, wheel folder and requirements file."""
    archives = sorted((OFFLINE / "python").glob("cpython-*.tar.gz"))
    wheels = OFFLINE / "wheels"
    requirements = OFFLINE / "requirements.txt"
    if (
        len(archives) != 1
        or not requirements.is_file()
        or not any(wheels.glob("*.whl"))
    ):
        raise SystemExit(
            f"{OFFLINE} must hold python/cpython-*.tar.gz (one), wheels/*.whl and "
            "requirements.txt; build the bundle with tools/pack_easy_run.py"
        )
    return archives[0], wheels, requirements


def probe_torch(python: str) -> dict:
    """What the given Python's PyTorch sees: version, CUDA and the first GPU."""
    probe = (
        "import json, torch; ok = torch.cuda.is_available(); "
        "p = torch.cuda.get_device_properties(0) if ok else None; "
        "print(json.dumps({'cuda': ok, 'torch': torch.__version__, "
        "'cuda_runtime': torch.version.cuda, "
        "'gpu': p.name if p else None, 'memory_gb': p.total_memory / 1e9 if p else 0}))"
    )
    result = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"PyTorch does not import in {python}:\n{result.stderr[-2000:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


# Every module the release path imports (training, evaluation, export and the
# runner). They are imported right after installation, so a package missing
# from the bundle stops the run within minutes instead of after days of
# training (export needs transformers, for example).
RELEASE_MODULES = (
    "torch", "numpy", "safetensors", "tokenizers", "tqdm", "tensorboard",
    "torch.utils.tensorboard", "transformers", "nuri.model", "nuri.modeling_nuri",
    "scripts.train", "scripts.evaluate", "scripts.export_release", "scripts.run_release",
)  # fmt: skip
IMPORT_CHECK = """
import importlib, sys
missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as error:
        missing.append(f"{name}: {type(error).__name__}: {error}")
print("\\n".join(missing))
sys.exit(1 if missing else 0)
"""


def check_imports(python: str, modules=RELEASE_MODULES) -> None:
    """Stop unless the given Python can import every release-path module."""
    path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [python, "-c", IMPORT_CHECK, *modules],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": path},
        capture_output=True, text=True,
    )  # fmt: skip
    if result.returncode != 0:
        raise SystemExit(
            "the installed environment cannot import what the release needs:\n"
            + (result.stdout.strip() or result.stderr[-2000:])
        )


def install(config: dict, state: dict) -> None:
    """Install the bundled Python and the pinned packages without network access.

    The bundled CPython (a relocatable python-build-standalone build) is
    unpacked into easy_run_work/python, a venv is created from it and pip
    installs offline/requirements.txt from offline/wheels with --no-index.
    The project itself is not installed: every command runs with the project
    folder on PYTHONPATH. Each part is skipped when it is already in place,
    so an interrupted install simply runs again.
    """
    archive, wheels, requirements = offline_files()
    python_home = WORK / "python"
    if not (python_home / "bin" / "python3").exists():
        staging = WORK / "python.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        check(["tar", "-xzf", str(archive), "-C", str(staging)])
        shutil.rmtree(python_home, ignore_errors=True)
        os.replace(staging / "python", python_home)
        shutil.rmtree(staging)
    venv_python = VENV / "bin" / "python"
    if not venv_python.exists():
        shutil.rmtree(VENV, ignore_errors=True)
        check([str(python_home / "bin" / "python3"), "-m", "venv", str(VENV)])
    check([str(venv_python), "-m", "pip", "install", "--no-index",
           "--find-links", str(wheels), "-r", str(requirements)])  # fmt: skip
    check_imports(str(venv_python))
    found = probe_torch(str(venv_python))
    if not found["cuda"]:
        raise SystemExit(
            f"PyTorch {found['torch']} does not see a CUDA GPU; check the NVIDIA driver "
            f"(it must support CUDA {found['cuda_runtime']})"
        )
    state["python"] = str(venv_python)
    state.setdefault("host", {}).update(found)
    use_python(state)
    log(f"installed: {found}")


def verify_data(config: dict, state: dict) -> None:
    """Check every bundled data file against its SHA256 list (once per list).

    The list is sha256sum format ("<sha256>  <path>"), so the operator can
    also run `sha256sum -c SHA256SUMS` inside the data folder by hand.
    """
    manifest = data_manifest(config)
    if not manifest.is_file():
        raise SystemExit(f"{manifest} is missing; unpack the whole bundle again")
    digest = file_sha256(manifest)
    if DATA_MARKER.exists() and DATA_MARKER.read_text().strip() == digest:
        log("the data was already verified")
        return
    folder = manifest.parent
    entries = [
        line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line
    ]
    log(f"verifying {len(entries)} data files")
    bad = []
    for expected, name in entries:
        path = folder / name.lstrip("*")
        if not path.is_file() or file_sha256(path) != expected:
            bad.append(name)
    if bad:
        raise SystemExit(
            f"{len(bad)} data files are missing or damaged (for example {bad[:3]}); "
            "unpack the bundle again"
        )
    missing = [path for path in input_paths(config) if not Path(path).exists()]
    if missing:
        raise SystemExit(f"the bundle data does not contain: {missing}")
    WORK.mkdir(parents=True, exist_ok=True)
    DATA_MARKER.write_text(digest)


def input_paths(config: dict) -> list[str]:
    """Every configured input and holdout path inside the unpacked data, without weights."""
    values = [
        config["inputs"][key]
        for key in ("tokenizer", "corpus", "instructions", "pairs")
    ]
    for split in config["holdouts"].values():
        values += list(split.values())
    values += list(config.get("evaluation", {}).values())
    flat = []
    for value in values:
        flat += value if isinstance(value, list) else [value]
    return [strip_weight(resolve(config, value)) for value in flat]


def strip_weight(value: str) -> str:
    """ "folder=0.6" -> "folder"; a plain path is returned unchanged."""
    path, separator, weight = value.rpartition("=")
    return path if separator and weight.replace(".", "", 1).isdigit() else value


def stage_data_flags(config: dict, kind: str) -> list[str]:
    inputs = config["inputs"]
    flags = ["--tokenizer", resolve(config, inputs["tokenizer"])]
    if kind in ("base", "it"):
        flags += ["--corpus", *resolve(config, inputs["corpus"])]
    if kind == "it":
        flags += ["--instructions", resolve(config, inputs["instructions"])]
    if kind == "embed":
        flags += ["--pairs", resolve(config, inputs["pairs"])]
    return flags


def scratch_init(preset: str) -> Path:
    """A randomly initialized checkpoint, so IT/embed timing needs no trained base."""
    folder = WORK / "calibration" / f"init-{preset}"
    if not (folder / "model.safetensors").exists():
        code = (
            "from scripts.train import PRESETS; from nuri.model import NuriModel; "
            "from nuri.checkpoint import save_pretrained; "
            f"save_pretrained(NuriModel(PRESETS[{preset!r}]()), {str(folder)!r})"
        )
        check([PYTHON, "-c", code])
    return folder


def seconds_per_step(log_file: Path) -> float:
    """Mean seconds per step after the first third (warmup and allocator growth).

    A mean, not a median, because base stages alternate cheaper MLM steps
    and dearer span steps.
    """
    rows = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    timed = [(row["step"], row["seconds"]) for row in rows if "seconds" in row]
    timed = timed[len(timed) // 3 :]
    if len(timed) < 2 or timed[-1][1] <= timed[0][1]:
        raise SystemExit(f"could not measure step time from {log_file}")
    (first_step, first_time), (last_step, last_time) = timed[0], timed[-1]
    return (last_time - first_time) / (last_step - first_step)


def peak_memory_share(log_file: Path, state: dict) -> float:
    """Largest reserved CUDA memory in a train log, as a share of the GPU (0 on CPU)."""
    total = state.get("host", {}).get("memory_gb", 0) * 1e9
    if not total:
        return 0.0
    rows = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    peak = max((row.get("cuda_peak_reserved_bytes", 0) for row in rows), default=0)
    return peak / total


def calibration_command(
    name: str, spec: dict, config: dict, micro: int, accum: int
) -> list[str]:
    out = WORK / "calibration" / name
    command = [PYTHON, "scripts/train.py", "--stage", spec["stage"], "--out", str(out),
               "--steps", str(CALIBRATION_STEPS), "--log-every", str(CALIBRATION_LOG_EVERY),
               "--no-progress", "--batch-size", str(micro), "--grad-accum", str(accum)]  # fmt: skip
    if spec["stage"] == "base":
        command += ["--preset", spec["preset"], "--pack-mlm", "--bucket"]
    else:
        command += ["--init", str(scratch_init(STAGES[spec["init"]]["preset"]))]
        if spec["stage"] == "it":
            command += ["--bucket"]
    return command + stage_data_flags(config, spec["stage"])


def calibrate(config: dict, state: dict) -> None:
    """Pick a micro-batch that fits and measure seconds per optimizer step, per stage.

    The effective batch (micro-batch x accumulation) stays as configured. A
    run that runs out of memory, or peaks above MEMORY_HEADROOM of the GPU,
    is repeated with half the micro-batch and twice the accumulation.
    """
    timings = state.setdefault("seconds_per_step", {})
    batches = state.setdefault("batches", {})
    for name, spec in STAGES.items():
        if name in timings:
            continue
        effective = batch_size(config, spec["stage"])
        micro = min(effective, MAX_MICRO_BATCH)
        out = WORK / "calibration" / name
        while True:
            shutil.rmtree(out, ignore_errors=True)
            accum = effective // micro
            command = calibration_command(name, spec, config, micro, accum)
            code, output = run_logged(command, f"calibrate-{name}")
            if code == INTERRUPTED_EXIT_CODE:
                raise SystemExit("calibration was stopped; run easy_run.py again")
            if code == 0:
                share = peak_memory_share(out / "train_log.jsonl", state)
                if share <= MEMORY_HEADROOM:
                    break
                log(f"{name}: micro-batch {micro} peaks at {share:.0%} of GPU memory")
            elif not is_out_of_memory(output):
                raise SystemExit(
                    f"calibration of {name} failed (exit {code}); see logs"
                )
            if micro // 2 < MIN_MICRO_BATCH:
                raise SystemExit(f"{name} does not fit even with micro-batch {micro}")
            micro //= 2
            log(f"{name}: retrying with micro-batch {micro}")
        timings[name] = seconds_per_step(out / "train_log.jsonl")
        batches[name] = {"micro_batch": micro, "grad_accum": accum}
        log(f"{name}: micro-batch {micro} x {accum}, {timings[name]:.3f} s/step")
        save_state(state)
        shutil.rmtree(out, ignore_errors=True)


def batch_size(config: dict, kind: str) -> int:
    train = config.get("train", {})
    return (
        train.get("embed_batch_size", 128)
        if kind == "embed"
        else train.get("batch_size", 64)
    )


def count_lines(path: str) -> int:
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def allocate_steps(config: dict, state: dict) -> dict:
    """Steps per stage from the remaining hours, the shares and the measured step times.

    IT and embedding stages are capped at budget.max_epochs passes over their
    data; hours freed by a cap go to base-dense, the main model.
    """
    budget = config["budget"]
    elapsed = (now() - state["started_at"]) / 3600
    # Evaluation and export of all five stages took under 10 minutes on the
    # A100 check (their sizes are fixed), and the deadline stop protects the
    # archive if the run is slower than planned, so the margins are small (D-027).
    reserve = budget.get("reserve_hours", 1)  # evaluation, export and packaging
    hours = max(0.0, budget["total_hours"] - elapsed - reserve) * budget.get(
        "safety", 0.95
    )
    shares = budget["shares"]
    total_share = sum(shares.values())
    stage_hours = {name: hours * shares[name] / total_share for name in STAGES}
    caps = {
        "it-dense": count_lines(resolve(config, config["inputs"]["instructions"])),
        "it-moe": count_lines(resolve(config, config["inputs"]["instructions"])),
        "embed": count_lines(resolve(config, config["inputs"]["pairs"])),
    }
    steps = {}
    for name in STAGES:
        seconds = state["seconds_per_step"][name]
        planned = int(stage_hours[name] * 3600 / seconds)
        if name in caps:
            kind = STAGES[name]["stage"]
            cap = math.ceil(
                budget.get("max_epochs", 3) * caps[name] / batch_size(config, kind)
            )
            if planned > cap:
                stage_hours["base-dense"] += (planned - cap) * seconds / 3600
                planned = cap
        steps[name] = planned
    steps["base-dense"] = int(
        stage_hours["base-dense"] * 3600 / state["seconds_per_step"]["base-dense"]
    )
    too_short = [
        name
        for name, count in steps.items()
        if count < (MIN_BASE_STEPS if STAGES[name]["stage"] == "base" else 1)
    ]
    if too_short:
        raise SystemExit(f"the budget leaves too few steps for {too_short}: {steps}")
    log(f"planned steps: {steps} (from {hours:.1f} training hours)")
    return steps


def write_holdout_config(config: dict) -> Path:
    holdouts = config["holdouts"]
    blocks = {}
    for split, seed in (("valid", 0), ("test", 1)):
        block = {"batches_per_task": 8, "seed": seed, "depth": 4}
        for key in ("corpus", "instructions", "pairs"):
            if holdouts[split].get(key):
                block[key] = resolve(config, holdouts[split][key])
        blocks[split] = block
    path = WORK / "holdouts.json"
    path.write_text(
        json.dumps({"schema_version": 1, "batch_size": 32, **blocks}, indent=2),
        encoding="utf-8",
    )
    return path


def write_recipes(config: dict) -> dict:
    """Optional per-stage-kind training recipes from easy_run.json "recipes"."""
    paths = {}
    for kind, corruption in config.get("recipes", {}).items():
        path = WORK / f"recipe-{kind}.json"
        recipe = {"schema_version": 1, "stage": kind, "corruption": corruption}
        path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        paths[kind] = str(path)
    return paths


def make_plan(config: dict, state: dict) -> None:
    steps = allocate_steps(config, state)
    holdouts = write_holdout_config(config)
    recipes = write_recipes(config)
    train = config.get("train", {})
    evaluation = config.get("evaluation", {})
    stages = {}
    for name, spec in STAGES.items():
        kind = spec["stage"]
        stage = {
            "stage": kind,
            "steps": steps[name],
            "batch_size": state["batches"][name]["micro_batch"],
            "grad_accum": state["batches"][name]["grad_accum"],
            "save_every_minutes": train.get("save_every_minutes", 30),
            "holdout_config": str(holdouts),
            "valid_every": max(
                1, steps[name] // train.get("validations_per_stage", 20)
            ),
            "release_name": spec["release_name"],
        }
        if EVALUATE[kind]:
            stage["evaluate"] = EVALUATE[kind]
        if kind == "it" and "reasoning_examples" in train:
            stage["reasoning_examples"] = train["reasoning_examples"]
        if "preset" in spec:
            stage.update(preset=spec["preset"], pack_mlm=True, bucket=True)
        if "init" in spec:
            stage["init"] = spec["init"]
        if kind == "it":
            stage["bucket"] = True
        if kind in ("it", "embed"):
            # These stages run up to max_epochs passes over a small set and can
            # overfit: they release the step with the best validation score and
            # end early once PATIENCE validations in a row bring no new best.
            stage["select_best"] = True
            stage["patience"] = PATIENCE
        if kind in recipes:
            stage["recipe_config"] = recipes[kind]
        stages[name] = stage
    inputs = config["inputs"]
    plan = {
        "out": str(WORK / "runs"),
        "data": {
            "tokenizer": resolve(config, inputs["tokenizer"]),
            "corpus": resolve(config, inputs["corpus"]),
            "instructions": resolve(config, inputs["instructions"]),
            "pairs": resolve(config, inputs["pairs"]),
            **{key: resolve(config, value) for key, value in evaluation.items()},
        },
        "release": {"gpu": state.get("host", {}).get("gpu") or "1x A100 80GB"},
        "train": {"tensorboard": True, "no_progress": True},
        "stages": stages,
    }
    path = WORK / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    state["plan"] = str(path)
    log(f"release plan written to {path}")


def deadline(config: dict, state: dict) -> float:
    """The now() time at which training must stop so packaging still fits."""
    budget = config["budget"]
    hours = budget["total_hours"] - budget.get("package_hours", 1.0)
    return state["started_at"] + hours * 3600


def train_all(config: dict, state: dict) -> None:
    """Run the release plan, restarting it after failures; every restart resumes.

    An out-of-memory failure turns on gradient checkpointing for the stages
    that are not finished yet (a resume-free setting: less memory, same
    results, about one extra forward pass). Other failures are retried as
    they are, because most are transient on a long run.

    At the deadline (budget.total_hours minus budget.package_hours after the
    first start) the running stage saves and stops, and easy_run moves on
    to packaging the stages that are finished. The plan keeps reserve
    hours, so this only happens when the run is slower than calibrated.
    """
    stop_at = deadline(config, state)
    for attempt in range(1, TRAIN_ATTEMPTS + 1):
        if now() >= stop_at:
            break
        code, output = run_logged(
            [PYTHON, "scripts/run_release.py", state["plan"]], "train", stop_at
        )
        if code == 0:
            return
        if now() >= stop_at:
            break  # stopped (or killed) at the deadline
        if code == INTERRUPTED_EXIT_CODE:
            raise SystemExit(
                "training stopped after saving; run `python easy_run.py` again to continue"
            )
        log(
            f"run_release.py failed with exit code {code} (attempt {attempt}/{TRAIN_ATTEMPTS})"
        )
        if is_out_of_memory(output):
            enable_gradient_checkpointing(Path(state["plan"]))
    else:
        raise SystemExit(
            "training kept failing; see easy_run_work/logs/train.log, then run again"
        )
    log("the time budget is used up; packaging the finished stages")
    state["deadline_reached"] = True


def enable_gradient_checkpointing(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = Path(plan["out"])
    for name, stage in plan["stages"].items():
        if not (runs / name / "final").exists():
            stage["gradient_checkpointing"] = True
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log("out of memory: gradient checkpointing is now on for unfinished stages")


def package(config: dict, state: dict, status: str = "complete") -> None:
    """Collect what should be brought back: models, evaluations, logs and summaries.

    Also called when easy_run stops early (status names the reason), so the
    archive always holds the finished stages and the logs that show what
    happened. status.json at the top of the archive says which.
    """
    runs = WORK / "runs"
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir()
    if (runs / "release").exists():
        shutil.copytree(runs / "release", OUTPUT / "models")
    if (runs / "eval").exists():
        shutil.copytree(runs / "eval", OUTPUT / "eval")
    for name in STAGES:
        stage_dir = runs / name
        if not stage_dir.exists():
            continue
        target = OUTPUT / "logs" / name
        target.mkdir(parents=True)
        for item in stage_dir.iterdir():
            if item.is_file() and item.suffix in (".json", ".jsonl"):
                shutil.copy2(item, target / item.name)
            elif item.is_dir() and item.name == "tensorboard":
                shutil.copytree(item, target / item.name)
    for item in (runs / "run_summary.json", STATE_PATH, LOG_PATH, WORK / "plan.json"):
        if item.exists():
            shutil.copy2(item, OUTPUT / item.name)
    if (WORK / "logs").exists():
        shutil.copytree(WORK / "logs", OUTPUT / "easy_run_logs")
    models = OUTPUT / "models"
    released = sorted(p.name for p in models.iterdir()) if models.exists() else []
    summary = {
        "status": status,
        "finished_steps": state.get("done", []),
        "deadline_reached": state.get("deadline_reached", False),
        "released_models": released,
    }
    (OUTPUT / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with tarfile.open(ARCHIVE, "w") as bundle:
        bundle.add(OUTPUT, arcname=OUTPUT.name)
    log(f"done: bring back {ARCHIVE.name} ({ARCHIVE.stat().st_size / 1e9:.2f} GB)")


STEPS = [
    ("preflight", preflight),
    ("install", install),
    ("data", verify_data),
    ("calibrate", calibrate),
    ("plan", make_plan),
    ("train", train_all),
    ("package", package),
]


def main() -> None:
    config = load_config()
    state = load_state()
    save_state(state)
    use_python(state)
    # The project runs from this folder without being installed.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])
    )
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"  # no network on the host
    log(f"easy_run started; finished steps: {state['done'] or 'none'}")
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        for name, step in STEPS:
            if name in state["done"]:
                continue
            log(f"step {name}")
            step(config, state)
            state["done"].append(name)
            save_state(state)
    except BaseException as error:
        # Whatever stopped the run, leave an archive with what is finished.
        save_state(state)
        if "package" not in state["done"]:
            reason = f"stopped early: {type(error).__name__}: {error}"
            log(reason)
            try:
                package(config, state, status=reason)
            except Exception as problem:  # never hide the original error
                log(f"packaging after the stop failed too: {problem}")
        raise
    log("all steps finished")


if __name__ == "__main__":
    main()
