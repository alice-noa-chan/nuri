import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from nuri.data import write_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>",
         "[UNK]"] + [f"w{i}" for i in range(120)]  # fmt: skip


def load_easy_run():
    spec = importlib.util.spec_from_file_location("easy_run", ROOT / "easy_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def easy(tmp_path, monkeypatch):
    module = load_easy_run()
    work = tmp_path / "work"
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "STATE_PATH", work / "state.json")
    monkeypatch.setattr(module, "LOG_PATH", work / "easy_run.log")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "easy_run_output")
    monkeypatch.setattr(module, "ARCHIVE", tmp_path / "easy_run_output.tar")
    monkeypatch.setattr(module, "CONFIG_PATH", tmp_path / "easy_run.json")
    monkeypatch.setattr(module, "OFFLINE", tmp_path / "offline")
    monkeypatch.setattr(module, "VENV", work / "venv")
    monkeypatch.setattr(module, "DATA_MARKER", work / "data_verified")
    monkeypatch.setattr(module, "PYTHON", sys.executable)
    return module


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def make_data(folder: Path) -> None:
    """A tiny but complete data folder: tokenizer, corpora and JSONL inputs per split."""
    tok = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (folder / "tok").mkdir(parents=True)
    tok.save(str(folder / "tok" / "tokenizer.json"))
    for split, offset in (("train", 0), ("valid", 3), ("test", 7)):
        write_corpus(folder / split / "corpus",
                     [[8 + (i + j + offset) % 100 for j in range(30)] for i in range(40)])  # fmt: skip
        write_jsonl(folder / split / "it.jsonl", [
            {"messages": [{"role": "user", "content": f"w{i + offset} w{i + 1}"}],
             "response": f"w{i + 2}"} for i in range(20)])  # fmt: skip
        write_jsonl(folder / split / "pairs.jsonl", [
            {"query": f"w{i + offset} w{i + 1}", "positive": f"w{i + 1} w{i + offset}"}
            for i in range(12)])  # fmt: skip


def write_manifest(folder: Path) -> None:
    """data/SHA256SUMS in sha256sum format, as tools/pack_easy_run.py writes it."""
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(folder).as_posix()}\n"
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (folder / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def make_config(data_folder="data", **budget) -> dict:
    return {
        "schema_version": 1,
        "data": {"folder": str(data_folder)},
        "inputs": {"tokenizer": "tok", "corpus": ["train/corpus=1"],
                   "instructions": "train/it.jsonl", "pairs": "train/pairs.jsonl"},
        "holdouts": {
            split: {"corpus": [f"{split}/corpus"], "instructions": f"{split}/it.jsonl",
                    "pairs": f"{split}/pairs.jsonl"}
            for split in ("valid", "test")
        },
        "evaluation": {"eval_corpus": "valid/corpus", "eval_pairs": "valid/pairs.jsonl"},
        "budget": {"total_hours": 0.05, "package_hours": 0, "safety": 1.0, "reserve_hours": 0,
                   "max_epochs": 2,
                   "shares": {"base-dense": 4, "base-moe": 2, "it-dense": 1, "it-moe": 1,
                              "embed": 1}, **budget},
        "train": {"batch_size": 4, "embed_batch_size": 4, "reasoning_examples": 0,
                  "validations_per_stage": 2},
        "recipes": {"base": {"span_end_prob": 0.5}},
    }  # fmt: skip


def test_config_validation_names_the_problem(easy, tmp_path):
    with pytest.raises(SystemExit, match="easy_run.json is missing"):
        easy.load_config()
    config = make_config("data")
    config["budget"]["shares"].pop("embed")
    easy.CONFIG_PATH.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="budget.shares"):
        easy.load_config()
    config = make_config("data", total_hours=1, package_hours=1)
    easy.CONFIG_PATH.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="package_hours"):
        easy.load_config()


def test_paths_resolve_inside_the_data_folder_and_keep_weights(easy):
    config = make_config("data")
    base = easy.ROOT / "data"
    assert easy.resolve(config, "train/corpus=0.6") == f"{base / 'train/corpus'}=0.6"
    assert easy.resolve(config, ["a", "b=2"]) == [str(base / "a"), f"{base / 'b'}=2"]
    assert easy.strip_weight("folder=0.25") == "folder"
    assert easy.strip_weight("odd=name") == "odd=name"


def test_step_time_uses_the_mean_after_warmup(easy, tmp_path):
    log = tmp_path / "train_log.jsonl"
    rows = [
        {"step": i, "seconds": t}
        for i, t in enumerate([0, 5, 5.5, 6, 7, 7.5, 8.5, 9, 10])
    ]
    write_jsonl(log, rows)
    # rows 3..8: (10 - 6) / (8 - 3)
    assert easy.seconds_per_step(log) == pytest.approx(0.8)


def test_allocation_caps_small_stages_and_gives_the_rest_to_base_dense(
    easy, monkeypatch, tmp_path
):
    make_data(tmp_path / "data")
    # A frozen clock: on a slow machine the real time between the start and
    # the planning call would already cost a step of base-moe.
    monkeypatch.setattr(easy, "now", lambda: 1000.0)
    config = make_config(tmp_path / "data", total_hours=10)
    state = {"started_at": easy.now(),
             "seconds_per_step": dict.fromkeys(easy.STAGES, 1.0)}  # fmt: skip
    steps = easy.allocate_steps(config, state)
    # IT: 2 epochs of 20 examples at batch 4; embed: 2 epochs of 12 pairs.
    assert steps["it-dense"] == steps["it-moe"] == 10 and steps["embed"] == 6
    assert steps["base-moe"] == int(10 * 3600 * 2 / 9)
    assert sum(steps.values()) == pytest.approx(10 * 3600, abs=3)
    config["budget"]["total_hours"] = 0.01
    with pytest.raises(SystemExit, match="too few steps"):
        easy.allocate_steps(config, state)


def test_end_to_end_on_cpu_with_tiny_presets(easy, monkeypatch, tmp_path):
    """Verify data, calibrate, plan, train every stage, evaluate, export and package."""
    stages = {name: dict(spec) for name, spec in easy.STAGES.items()}
    stages["base-dense"]["preset"] = stages["base-moe"]["preset"] = "tiny"
    monkeypatch.setattr(easy, "STAGES", stages)
    monkeypatch.setattr(
        easy, "EVALUATE", {"base": ["mlm"], "it": [], "embed": ["retrieval"]}
    )
    monkeypatch.setattr(easy, "CALIBRATION_STEPS", 6)
    monkeypatch.setattr(easy, "MIN_BASE_STEPS", 2)
    monkeypatch.setattr(
        easy, "STEPS", [s for s in easy.STEPS if s[0] not in ("preflight", "install")]
    )
    make_data(tmp_path / "data")
    write_manifest(tmp_path / "data")
    monkeypatch.setattr(
        easy, "now", lambda: 0.0
    )  # the test's own runtime is not budget
    # The frozen clock never reaches a deadline, but the stop timer runs in
    # real seconds; keep it far away.
    monkeypatch.setattr(easy, "deadline", lambda config, state: 1e5)
    config = make_config(tmp_path / "data", total_hours=0.01)
    easy.CONFIG_PATH.write_text(json.dumps(config))

    easy.main()

    state = json.loads(easy.STATE_PATH.read_text())
    assert state["done"] == ["data", "calibrate", "plan", "train", "package"]
    assert set(state["seconds_per_step"]) == set(stages)
    plan = json.loads((easy.WORK / "plan.json").read_text())
    assert plan["stages"]["base-dense"]["recipe_config"].endswith("recipe-base.json")
    assert plan["stages"]["it-dense"]["init"] == "base-dense"
    output = easy.OUTPUT
    for name in (
        "nuri-base",
        "nuri-base-moe",
        "nuri-base-it",
        "nuri-base-moe-it",
        "nuri-base-embed",
    ):
        assert (output / "models" / name / "model.safetensors").is_file()
    assert (output / "eval" / "base-dense.json").is_file()
    assert (output / "logs" / "base-dense" / "train_log.jsonl").is_file()
    assert (output / "logs" / "base-dense" / "holdout_log.jsonl").is_file()
    with tarfile.open(easy.ARCHIVE) as bundle:
        assert "easy_run_output/models/nuri-base/model.safetensors" in bundle.getnames()
    assert easy.DATA_MARKER.is_file()  # the data is checked only once

    # A second run finds every step finished and does nothing.
    easy.main()
    assert "finished steps: ['data'" in easy.LOG_PATH.read_text().splitlines()[-2]


def test_example_config_passes_validation_once_filled_in(easy):
    config = json.loads((ROOT / "configs" / "easy_run.example.json").read_text())
    easy.CONFIG_PATH.write_text(json.dumps(config))
    loaded = easy.load_config()
    assert set(loaded["budget"]["shares"]) == set(easy.STAGES)
    assert loaded["budget"]["total_hours"] == 168


def fake_calibration(easy, monkeypatch, outcomes):
    """Replace run_logged: each call pops (exit code, output, peak bytes) and writes a log."""
    commands = []

    def run_logged(command, name):
        commands.append(command)
        code, output, peak = outcomes.pop(0)
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"step": i, "seconds": float(i), "cuda_peak_reserved_bytes": peak}
                for i in range(easy.CALIBRATION_STEPS)]  # fmt: skip
        write_jsonl(out / "train_log.jsonl", rows)
        return code, output

    monkeypatch.setattr(easy, "run_logged", run_logged)
    monkeypatch.setattr(easy, "scratch_init", lambda preset: Path("init"))
    return commands


def test_calibration_halves_the_micro_batch_on_oom_or_tight_memory(easy, monkeypatch):
    monkeypatch.setattr(
        easy,
        "STAGES",
        {"embed": easy.STAGES["embed"], "base-dense": easy.STAGES["base-dense"]},
    )
    config = make_config("data")
    config["train"]["embed_batch_size"] = 128
    config["train"]["batch_size"] = 64
    gb = 1e9
    outcomes = [
        (1, "torch.OutOfMemoryError: CUDA out of memory", 0),  # embed micro 64
        (0, "", 79 * gb),  # embed micro 32: fits but peaks above the headroom
        (0, "", 40 * gb),  # embed micro 16
        (0, "", 50 * gb),  # base-dense micro 64
    ]
    commands = fake_calibration(easy, monkeypatch, outcomes)
    state = {"host": {"memory_gb": 80}}
    easy.calibrate(config, state)
    micro = [c[c.index("--batch-size") + 1] for c in commands]
    accum = [c[c.index("--grad-accum") + 1] for c in commands]
    assert micro == ["64", "32", "16", "64"] and accum == ["2", "4", "8", "1"]
    assert state["batches"] == {"embed": {"micro_batch": 16, "grad_accum": 8},
                                "base-dense": {"micro_batch": 64, "grad_accum": 1}}  # fmt: skip
    assert state["seconds_per_step"]["embed"] == pytest.approx(1.0)


def test_calibration_stops_on_other_failures(easy, monkeypatch):
    monkeypatch.setattr(easy, "STAGES", {"base-dense": easy.STAGES["base-dense"]})
    fake_calibration(easy, monkeypatch, [(1, "ValueError: bad corpus", 0)])
    with pytest.raises(SystemExit, match="calibration of base-dense failed"):
        easy.calibrate(make_config("data"), {"host": {}})


def write_plan(easy, finished=("base-dense",)) -> Path:
    runs = easy.WORK / "runs"
    for name in finished:
        (runs / name / "final").mkdir(parents=True)
    plan_path = easy.WORK / "plan.json"
    plan_path.write_text(
        json.dumps({"out": str(runs), "stages": {"base-dense": {}, "it-dense": {}}})
    )
    return plan_path


def fake_runs(easy, monkeypatch, results):
    """Replace run_logged with a queue of (exit code, output) answers."""
    monkeypatch.setattr(
        easy, "run_logged", lambda command, name, stop_at=None: results.pop(0)
    )


def test_training_retries_and_turns_on_checkpointing_after_oom(easy, monkeypatch):
    easy.WORK.mkdir(parents=True)
    plan_path = write_plan(easy)
    config = make_config("data", total_hours=10)
    state = {"plan": str(plan_path), "started_at": easy.now()}
    results = [(1, "CUDA out of memory"), (0, "")]
    fake_runs(easy, monkeypatch, results)
    easy.train_all(config, state)
    stages = json.loads(plan_path.read_text())["stages"]
    assert stages["it-dense"]["gradient_checkpointing"] is True
    assert "gradient_checkpointing" not in stages["base-dense"]

    results.extend([(1, "boom")] * easy.TRAIN_ATTEMPTS)
    with pytest.raises(SystemExit, match="kept failing"):
        easy.train_all(config, state)
    fake_runs(easy, monkeypatch, [(75, "")])
    with pytest.raises(SystemExit, match="run `python easy_run.py` again"):
        easy.train_all(config, state)


def test_training_stops_at_the_deadline_and_moves_on(easy, monkeypatch):
    easy.WORK.mkdir(parents=True)
    plan_path = write_plan(easy)
    config = make_config("data", total_hours=10, package_hours=1)
    clock = {"now": 0.0}
    monkeypatch.setattr(easy, "now", lambda: clock["now"])
    state = {"plan": str(plan_path), "started_at": 0.0}
    assert easy.deadline(config, state) == 9 * 3600
    seen = []

    def run_logged(command, name, stop_at=None):
        seen.append(stop_at)
        clock["now"] = stop_at  # the run lasts until the deadline stops it
        return 75, ""

    monkeypatch.setattr(easy, "run_logged", run_logged)
    easy.train_all(config, state)  # no error: packaging comes next
    assert seen == [9 * 3600] and state["deadline_reached"] is True

    # Restarted after the deadline: nothing is trained at all.
    seen.clear()
    easy.train_all(config, {"plan": str(plan_path), "started_at": 0.0})
    assert seen == []


def test_run_logged_stops_a_child_at_the_deadline(easy, monkeypatch):
    monkeypatch.setattr(easy, "STOP_GRACE_SECONDS", 30)
    sleeper = [
        sys.executable,
        "-c",
        "import time; print('up', flush=True); time.sleep(60)",
    ]
    started = easy.time.monotonic()
    code, output = easy.run_logged(sleeper, "sleeper", stop_at=easy.now() + 1)
    assert code != 0 and "up" in output
    assert easy.time.monotonic() - started < 30
    assert "the time budget is used up" in easy.LOG_PATH.read_text()
    assert easy._child is None


def test_sigterm_is_passed_to_the_running_child(easy):
    easy.WORK.mkdir(parents=True)
    child = easy.subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    easy._child = child
    try:
        easy.on_sigterm(easy.signal.SIGTERM, None)
        assert child.wait(timeout=30) != 0
    finally:
        easy._child = None
        child.kill()
    with pytest.raises(SystemExit, match="received SIGTERM"):
        easy.on_sigterm(easy.signal.SIGTERM, None)


def test_a_failed_run_still_packages_what_is_finished(easy, monkeypatch):
    model = easy.WORK / "runs" / "release" / "nuri-base"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"weights")
    (easy.WORK / "logs").mkdir()
    (easy.WORK / "logs" / "train.log").write_text("it-dense crashed")
    easy.CONFIG_PATH.write_text(json.dumps(make_config("data")))

    def broken(config, state):
        raise SystemExit("training kept failing")

    monkeypatch.setattr(easy, "STEPS", [("train", broken), ("package", easy.package)])
    with pytest.raises(SystemExit, match="kept failing"):
        easy.main()
    with tarfile.open(easy.ARCHIVE) as bundle:
        names = bundle.getnames()
        status = json.load(bundle.extractfile("easy_run_output/status.json"))
    assert "easy_run_output/models/nuri-base/model.safetensors" in names
    assert "easy_run_output/easy_run_logs/train.log" in names
    assert status["status"].startswith("stopped early: SystemExit: training kept")
    assert status["released_models"] == ["nuri-base"]
    state = json.loads(easy.STATE_PATH.read_text())
    assert state["done"] == []  # the next run tries training again


def test_batches_must_be_powers_of_two(easy):
    config = make_config("data")
    config["train"]["batch_size"] = 48
    easy.CONFIG_PATH.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="power of two"):
        easy.load_config()


def test_dense_products_train_before_the_moe_variants(easy):
    order = list(easy.STAGES)
    assert order[:3] == ["base-dense", "it-dense", "embed"]
    for name, spec in easy.STAGES.items():
        if "init" in spec:  # every stage starts after the stage it builds on
            assert order.index(spec["init"]) < order.index(name)


def test_data_is_checked_against_the_bundled_list_once(easy, monkeypatch, tmp_path):
    folder = tmp_path / "data"
    make_data(folder)
    write_manifest(folder)
    config = make_config(folder)
    easy.verify_data(config, {})
    assert easy.DATA_MARKER.is_file()

    hashed = []
    monkeypatch.setattr(easy, "file_sha256", lambda path: hashed.append(path) or "x")
    easy.DATA_MARKER.write_text("x")  # the marker matches the (fake) list digest
    easy.verify_data(config, {})
    assert len(hashed) == 1  # only the list itself, not every data file
    monkeypatch.undo()


def test_damaged_or_missing_data_stops_with_a_clear_message(easy, tmp_path):
    folder = tmp_path / "data"
    make_data(folder)
    write_manifest(folder)
    config = make_config(folder)
    (folder / "train" / "it.jsonl").write_text("changed in transfer")
    with pytest.raises(SystemExit, match="1 data files are missing or damaged"):
        easy.verify_data(config, {})
    assert not easy.DATA_MARKER.exists()
    (folder / "SHA256SUMS").unlink()
    with pytest.raises(SystemExit, match="SHA256SUMS is missing"):
        easy.verify_data(config, {})


def make_offline(folder: Path) -> None:
    (folder / "python").mkdir(parents=True)
    (
        folder
        / "python"
        / "cpython-3.11.0-x86_64-unknown-linux-gnu-install_only.tar.gz"
    ).write_bytes(b"")
    (folder / "wheels").mkdir()
    (folder / "wheels" / "torch-0-py3-none-any.whl").write_bytes(b"")
    (folder / "requirements.txt").write_text("torch\n")


def test_install_uses_only_the_bundled_python_and_wheels(easy, monkeypatch):
    make_offline(easy.OFFLINE)
    commands = []

    def check(command):
        commands.append(command)
        if command[0] == "tar":  # unpack: python-build-standalone has python/bin
            (Path(command[-1]) / "python" / "bin").mkdir(parents=True)
            (Path(command[-1]) / "python" / "bin" / "python3").write_text("")
        if command[1:3] == ["-m", "venv"]:
            (Path(command[-1]) / "bin").mkdir(parents=True)
            (Path(command[-1]) / "bin" / "python").write_text("")

    found = {"cuda": True, "torch": "2.10.0+cu128", "cuda_runtime": "12.8",
             "gpu": "NVIDIA A100-SXM4-80GB", "memory_gb": 85.0}  # fmt: skip
    monkeypatch.setattr(easy, "check", check)
    monkeypatch.setattr(easy, "probe_torch", lambda python: found)
    state = {}
    easy.install(make_config("data"), state)
    assert commands[0][:2] == ["tar", "-xzf"]
    assert commands[1][1:] == ["-m", "venv", str(easy.VENV)]
    pip = commands[2]
    assert pip[1:4] == ["-m", "pip", "install"] and "--no-index" in pip
    assert pip[pip.index("--find-links") + 1] == str(easy.OFFLINE / "wheels")
    assert state["python"] == str(easy.VENV / "bin" / "python") == easy.PYTHON
    assert state["host"]["torch"] == "2.10.0+cu128"

    # A second install (after an interruption) only reruns pip.
    commands.clear()
    easy.install(make_config("data"), state)
    assert len(commands) == 1 and commands[0] == pip

    found["cuda"] = False
    with pytest.raises(SystemExit, match="does not see a CUDA GPU"):
        easy.install(make_config("data"), {})


def test_preflight_reads_the_gpu_from_nvidia_smi(easy, monkeypatch):
    make_offline(easy.OFFLINE)

    class Result:
        returncode, stderr = 0, ""
        stdout = "NVIDIA A100-SXM4-80GB, 81920, 570.86\n"

    monkeypatch.setattr(easy.subprocess, "run", lambda *a, **k: Result())
    state = {}
    easy.preflight(make_config("data"), state)
    assert state["host"]["gpu"] == "NVIDIA A100-SXM4-80GB"
    assert state["host"]["memory_gb"] == pytest.approx(85.9, abs=0.1)
    assert state["host"]["driver"] == "570.86"

    (easy.OFFLINE / "requirements.txt").unlink()
    with pytest.raises(SystemExit, match="requirements.txt"):
        easy.preflight(make_config("data"), {})
