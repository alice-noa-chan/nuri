"""Stopping and saving safely when a GPU host is preempted.

Covers atomic checkpoint writes, the SIGTERM stop path (exercised through
request_stop, because sending SIGTERM to the test process would end it on
Windows), timed saves and the runner's handling of a stopped stage.
"""

import json
import random
import sys
from pathlib import Path

import pytest
import torch

from nuri import checkpoint
from nuri.config import tiny_config
from nuri.data import TokenCorpus, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage
from nuri.trainer import (INTERRUPTED_EXIT_CODE, RESUME_FREE_SETTINGS, Task, TrainingInterrupted, TrainSettings,
                          request_stop, train)

CONFIG = tiny_config()


def toy_corpus(folder, count=120):
    rng = random.Random(0)
    docs = [[3 + (rng.randint(3, 60) + i) % 120 for i in range(rng.randint(20, 40))] for _ in range(count)]
    write_corpus(folder, docs)
    return TokenCorpus(folder)


def fresh(corpus, out, steps=8, **settings):
    torch.manual_seed(0)
    model = NuriModel(CONFIG)
    tasks, mix = base_stage(corpus, CONFIG, batch_size=4, seed=3)
    return model, tasks, TrainSettings(out_dir=str(out), total_steps=steps, peak_lr=3e-3, warmup_steps=1,
                                       log_every=1, mix=mix, progress=False, **settings)


def stop_after(tasks: dict, calls: int) -> dict:
    """Wrap every task so that the `calls`-th batch (counted over all tasks) requests a stop."""
    count = {"n": 0}

    def wrap(task: Task) -> Task:
        def next_batch():
            count["n"] += 1
            if count["n"] == calls:
                request_stop()
            return task.next_batch()
        return Task(task.kind, next_batch, task.rng, task.generator, task.options)

    return {name: wrap(task) for name, task in tasks.items()}


def test_failed_write_keeps_the_previous_complete_file(tmp_path):
    path = tmp_path / "state.pt"
    checkpoint.atomic_write(path, lambda temporary: torch.save({"step": 1}, temporary))

    def broken(temporary):
        temporary.write_bytes(b"partial")
        raise OSError("disk went away")

    with pytest.raises(OSError):
        checkpoint.atomic_write(path, broken)
    assert torch.load(path)["step"] == 1
    assert list(tmp_path.iterdir()) == [path]  # no temporary file is left behind


def test_stop_request_saves_latest_and_resumes_exactly(tmp_path):
    corpus = toy_corpus(tmp_path / "corpus")

    model, tasks, settings = fresh(corpus, tmp_path / "full")
    train(model, tasks, settings, device=torch.device("cpu"))

    stopped, tasks, settings = fresh(corpus, tmp_path / "stopped")
    with pytest.raises(TrainingInterrupted) as info:
        train(stopped, stop_after(tasks, 3), settings, device=torch.device("cpu"))
    assert info.value.step == 3 and info.value.path.name == "latest.pt"
    assert not (tmp_path / "stopped" / "final").exists()

    resumed, tasks, settings = fresh(corpus, tmp_path / "stopped")
    records = train(resumed, tasks, settings, device=torch.device("cpu"), resume_from=info.value.path)
    assert records[0]["step"] == 3
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name]), name


def test_stop_on_the_last_step_just_finishes(tmp_path):
    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "run", steps=3)
    assert train(model, stop_after(tasks, 3), settings, device=torch.device("cpu"))
    assert (tmp_path / "run" / "final" / "model.safetensors").exists()
    assert not (tmp_path / "run" / "training_state" / "latest.pt").exists()


def test_timed_saves_write_latest_and_do_not_block_resume(tmp_path):
    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "run", save_every_minutes=1e-9)
    train(model, tasks, settings, device=torch.device("cpu"))
    latest = torch.load(tmp_path / "run" / "training_state" / "latest.pt", weights_only=False)
    # Timed saves happen after steps without a regular checkpoint (which resets the
    # timer instead) and never after the final step.
    assert 1 <= latest["step"] < settings.total_steps
    assert "save_every_minutes" in RESUME_FREE_SETTINGS

    # The timed-save interval may change between the interrupted and the resumed run.
    resumed, tasks, settings = fresh(corpus, tmp_path / "again", save_every_minutes=0.0)
    assert train(resumed, tasks, settings, device=torch.device("cpu"),
                 resume_from=tmp_path / "run" / "training_state" / "latest.pt")


def _runner(tmp_path, monkeypatch, codes, on_wait=None):
    """Load scripts/run_release.py with a fake child process returning `codes` in order."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_release

    plan = {"out": str(tmp_path / "out"), "data": {"corpus": "c"},
            "stages": {"base-dense": {"stage": "base", "steps": 4, "evaluate": ["mlm"], "release_name": "x"},
                       "it-dense": {"stage": "it", "init": "base-dense", "steps": 4}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    calls, codes = [], list(codes)

    class FakeChild:
        def __init__(self, command, cwd):  # noqa: ARG002
            calls.append(command)
            self.pid = -1

        def wait(self):
            if on_wait is not None:
                on_wait()
            return codes.pop(0)

    killed = []
    monkeypatch.setattr(run_release.subprocess, "Popen", FakeChild)
    monkeypatch.setattr(run_release.os, "kill", lambda pid, signum: killed.append((pid, signum)))
    run_release.killed = killed
    return run_release, path, calls


def test_runner_stops_the_plan_on_an_interrupted_stage(tmp_path, monkeypatch):
    run_release, path, calls = _runner(tmp_path, monkeypatch, [INTERRUPTED_EXIT_CODE])
    assert run_release.INTERRUPTED_EXIT_CODE == INTERRUPTED_EXIT_CODE
    with pytest.raises(SystemExit) as info:
        run_release.main([str(path)])
    assert info.value.code == INTERRUPTED_EXIT_CODE
    assert len(calls) == 1  # no evaluation, export or later stage after the stop
    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["base-dense"]["train_seconds"] >= 0

    run_release, path, calls = _runner(tmp_path, monkeypatch, [1])
    with pytest.raises(run_release.subprocess.CalledProcessError):
        run_release.main([str(path)])


def test_runner_forwards_sigterm_to_the_running_step(tmp_path, monkeypatch):
    import signal

    # The child "receives" SIGTERM from the runner and dies with -15 (a POSIX kill);
    # the runner must report a stop (75), not a failure.
    run_release, path, calls = _runner(
        tmp_path, monkeypatch, [-15],
        on_wait=lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None))
    with pytest.raises(SystemExit) as info:
        run_release.main([str(path)])
    assert info.value.code == INTERRUPTED_EXIT_CODE and len(calls) == 1
    assert run_release.killed == [(-1, signal.SIGTERM)]


def test_signal_handlers_take_no_locks():
    """A re-entered handler must not block: both handlers only set a flag (and kill a pid)."""
    import signal

    from nuri import trainer

    trainer._stop.requested = False
    trainer._on_sigterm(signal.SIGTERM, None)
    trainer._on_sigterm(signal.SIGTERM, None)  # a second signal is harmless
    assert trainer._stop.requested is True
    trainer._stop.requested = False
    assert not hasattr(trainer, "_stop_requested")  # the lock-based Event is gone


def test_runner_needs_the_final_state_before_skipping_a_stage(tmp_path, monkeypatch):
    def write_evaluation_result():
        command = calls[-1]
        if "evaluate.py" in command[1]:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"status": "synthetic_fixture"}), encoding="utf-8")

    run_release, path, calls = _runner(tmp_path, monkeypatch, [0, 0, 0, 0], on_wait=write_evaluation_result)
    monkeypatch.setattr(run_release.platform, "platform", lambda: "synthetic-platform")
    plan = json.loads(path.read_text(encoding="utf-8"))
    toy_corpus(tmp_path / "heldout")
    plan["data"]["eval_corpus"] = str(tmp_path / "heldout")
    path.write_text(json.dumps(plan), encoding="utf-8")
    stage = tmp_path / "out" / "base-dense"
    (stage / "final").mkdir(parents=True)
    (stage / "final" / "model.safetensors").write_bytes(b"weights")
    CONFIG.save(stage / "final" / "config.json")
    run_release.main([str(path)])
    assert "--stage" in calls[0] and "base" in calls[0]  # trained again: final.pt was missing

    (stage / "training_state").mkdir()
    (stage / "training_state" / "final.pt").write_bytes(b"state")
    calls.clear()
    run_release, path, calls = _runner(tmp_path, monkeypatch, [0, 0, 0], on_wait=write_evaluation_result)
    path.write_text(json.dumps(plan), encoding="utf-8")
    run_release.main([str(path)])
    assert "export_release.py" in calls[0][1]  # completed training and bound evaluation were both reused
    assert not any(Path(command[1]).name == "train.py" and "--stage" in command
                   and command[command.index("--stage") + 1] == "base" for command in calls)


def test_final_weights_appear_only_after_the_final_state(tmp_path, monkeypatch):
    from nuri import trainer

    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "run", steps=3)
    original = trainer.save_training_state

    def failing(path, *args, **kwargs):
        if Path(path).name == "final.pt":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(trainer, "save_training_state", failing)
    with pytest.raises(OSError):
        train(model, tasks, settings, device=torch.device("cpu"))
    assert not (tmp_path / "run" / "final" / "model.safetensors").exists()


def test_resuming_a_finished_state_restores_missing_final_weights(tmp_path):
    from nuri.checkpoint import load_pretrained

    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "run", steps=3)
    train(model, tasks, settings, device=torch.device("cpu"))
    weights = tmp_path / "run" / "final" / "model.safetensors"
    expected = load_pretrained(tmp_path / "run" / "final").state_dict()
    weights.unlink()

    again, tasks, settings = fresh(corpus, tmp_path / "run", steps=3)
    assert train(again, tasks, settings, device=torch.device("cpu"),
                 resume_from=tmp_path / "run" / "training_state" / "final.pt") == []
    restored = load_pretrained(tmp_path / "run" / "final").state_dict()
    assert all(torch.equal(expected[name], restored[name]) for name in expected)


def test_sigterm_handler_is_installed_only_while_training(tmp_path):
    import signal

    from nuri import trainer

    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "run", steps=2)
    seen = []
    task = tasks["mlm"]
    tasks["mlm"] = Task(task.kind, lambda: (seen.append(signal.getsignal(signal.SIGTERM)), task.next_batch())[1],
                        task.rng, task.generator, task.options)
    before = signal.getsignal(signal.SIGTERM)
    train(model, tasks, settings, device=torch.device("cpu"))
    assert seen and all(handler is trainer._on_sigterm for handler in seen)
    assert signal.getsignal(signal.SIGTERM) is before


def test_stop_at_a_checkpoint_boundary_writes_one_state_and_completes_on_resume(tmp_path, monkeypatch):
    from nuri import trainer
    from nuri.checkpoint import load_pretrained

    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "full", save_every=3)
    train(model, tasks, settings, device=torch.device("cpu"))

    written = []
    original = trainer.save_training_state
    monkeypatch.setattr(trainer, "save_training_state",
                        lambda path, *a, **k: (written.append(Path(path).name), original(path, *a, **k))[1])
    stopped, tasks, settings = fresh(corpus, tmp_path / "stopped", save_every=3)
    with pytest.raises(TrainingInterrupted) as info:  # the stop arrives during step 3, a step-3 boundary
        train(stopped, stop_after(tasks, 3), settings, device=torch.device("cpu"))
    assert info.value.step == 3 and written == ["latest.pt"]  # one full-state save, nothing else
    assert not (tmp_path / "stopped" / "step-3").exists()

    resumed, tasks, settings = fresh(corpus, tmp_path / "stopped", save_every=3)
    train(resumed, tasks, settings, device=torch.device("cpu"), resume_from=info.value.path)
    for name in ("step-3", "final"):
        expected = load_pretrained(tmp_path / "full" / name).state_dict()
        actual = load_pretrained(tmp_path / "stopped" / name).state_dict()
        assert all(torch.equal(expected[k], actual[k]) for k in expected), name
    assert (tmp_path / "stopped" / "training_state" / "step-3.pt").exists()


def _nonfinite_after(trainer, monkeypatch, calls: int, where: str):
    """From the `calls`-th optimizer step on, make the loss NaN or the gradient norm inf."""
    count = {"n": 0}
    if where == "loss":
        original = trainer._task_loss

        def task_loss(*args, **kwargs):
            result = original(*args, **kwargs)
            count["n"] += 1
            if count["n"] >= calls:
                result["loss"] = result["loss"] * float("nan")
            return result
        monkeypatch.setattr(trainer, "_task_loss", task_loss)
    else:
        original = torch.nn.utils.clip_grad_norm_

        def clip(*args, **kwargs):
            norm = original(*args, **kwargs)
            count["n"] += 1
            return torch.tensor(float("inf")) if count["n"] >= calls else norm
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)


@pytest.mark.parametrize("where", ["loss", "gradient"])
def test_nonfinite_values_stop_before_the_update_and_save_nothing(tmp_path, monkeypatch, where):
    from nuri import trainer

    corpus = toy_corpus(tmp_path / "corpus")
    model, tasks, settings = fresh(corpus, tmp_path / "run", save_every=1, save_every_minutes=1e-9)
    _nonfinite_after(trainer, monkeypatch, 3, where)
    request_stop_on = stop_after(tasks, 3)  # a stop request arrives on the same step

    states = tmp_path / "run" / "training_state"
    with pytest.raises(trainer.NonFiniteTraining) as info:
        train(model, request_stop_on, settings, device=torch.device("cpu"))
    assert info.value.step == 2
    assert sorted(p.name for p in states.iterdir()) == ["step-1.pt", "step-2.pt"]  # no latest.pt, no step-3
    assert all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())
    assert trainer.NONFINITE_EXIT_CODE == 3


def test_runner_summary_is_atomic_and_counts_failed_attempts(tmp_path, monkeypatch):
    run_release, path, calls = _runner(tmp_path, monkeypatch, [1])
    with pytest.raises(run_release.subprocess.CalledProcessError):
        run_release.main([str(path)])
    summary_path = tmp_path / "out" / "run_summary.json"
    first = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "train_seconds" in first["base-dense"]  # the failed attempt was recorded

    def broken_replace(source, target):
        raise OSError("killed during the rename")

    monkeypatch.setattr(run_release.os, "replace", broken_replace)
    with pytest.raises(OSError):
        run_release.write_summary(summary_path, {"broken": True})
    assert json.loads(summary_path.read_text(encoding="utf-8")) == first  # the previous summary survives
