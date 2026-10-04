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
    assert latest["step"] == settings.total_steps - 1  # the last timed save, before the final step
    assert "save_every_minutes" in RESUME_FREE_SETTINGS

    # The timed-save interval may change between the interrupted and the resumed run.
    resumed, tasks, settings = fresh(corpus, tmp_path / "again", save_every_minutes=0.0)
    assert train(resumed, tasks, settings, device=torch.device("cpu"),
                 resume_from=tmp_path / "run" / "training_state" / "latest.pt")


def test_runner_stops_the_plan_on_an_interrupted_stage(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_release

    assert run_release.INTERRUPTED_EXIT_CODE == INTERRUPTED_EXIT_CODE
    plan = {"out": str(tmp_path / "out"), "data": {"corpus": "c"},
            "stages": {"base-dense": {"stage": "base", "steps": 4, "evaluate": ["mlm"], "release_name": "x"},
                       "it-dense": {"stage": "it", "init": "base-dense", "steps": 4}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    calls = []

    class Finished:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, cwd):  # noqa: ARG001
        calls.append(command)
        return Finished(INTERRUPTED_EXIT_CODE)

    monkeypatch.setattr(run_release.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as info:
        run_release.main([str(path)])
    assert info.value.code == INTERRUPTED_EXIT_CODE
    assert len(calls) == 1  # no evaluation, export or later stage after the stop
    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["base-dense"]["train_seconds"] >= 0

    monkeypatch.setattr(run_release.subprocess, "run", lambda command, cwd: Finished(1))
    with pytest.raises(run_release.subprocess.CalledProcessError):
        run_release.main([str(path)])


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
