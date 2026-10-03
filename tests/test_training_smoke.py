"""End-to-end CPU smoke test: base -> instruction tuning -> embedding on toy data."""

import json
import random

import torch

from nuri.checkpoint import load_pretrained
from nuri.config import MoEConfig, tiny_config
from nuri.data import TokenCorpus, build_instruction_batch, write_corpus
from nuri.generation import generate
from nuri.model import NuriModel
from nuri.pipeline import base_stage, embedding_stage, instruction_stage
from nuri.trainer import TrainSettings, train


def toy_corpus(folder, count=200):
    """Documents that count upward from a random start: easy to learn."""
    rng = random.Random(0)
    docs = []
    for _ in range(count):
        start = rng.randint(3, 60)
        docs.append([3 + (start + i) % 120 for i in range(rng.randint(20, 60))])
    write_corpus(folder, docs)
    return TokenCorpus(folder)


def run(model, tasks, mix, out, steps, lr=3e-3):
    settings = TrainSettings(out_dir=str(out), total_steps=steps, peak_lr=lr, warmup_steps=5,
                             log_every=1, save_every=10, mix=mix)
    return train(model, tasks, settings, device=torch.device("cpu"))


def test_all_three_stages(tmp_path):
    torch.manual_seed(0)
    config = tiny_config(moe=MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,),
                                       readout_layers=(2,)))
    corpus = toy_corpus(tmp_path / "corpus")

    # Base stage: loss must go down and checkpoints must appear.
    model = NuriModel(config)
    tasks, mix = base_stage(corpus, config, batch_size=8)
    records = run(model, tasks, mix, tmp_path / "base", steps=40)
    first = [r["loss"] for r in records[:8]]
    last = [r["loss"] for r in records[-8:]]
    assert sum(last) < sum(first)
    assert {r["task"] for r in records} == {"mlm", "span"}
    assert "moe_balance" in records[0] and "expert_counts" in records[0]
    for name in ("step-10", "pre-decay", "final"):
        assert (tmp_path / "base" / name / "model.safetensors").exists()
    assert (tmp_path / "base" / "training_state" / "final.pt").exists()
    log_lines = (tmp_path / "base" / "train_log.jsonl").read_text().splitlines()
    assert len(log_lines) == 40 and json.loads(log_lines[-1])["step"] == 39

    # Instruction tuning from the base checkpoint; the length head becomes usable.
    model = load_pretrained(tmp_path / "base" / "final")
    model.config.length_head_trained = True
    examples = [([5, 6, 7], [8, 9]), ([10, 11], [12, 13, 14])] * 8
    reasoning = [([20, 21], [22])] * 8
    tasks, mix = instruction_stage(examples, model.config, 4, corpus, reasoning)
    assert mix == {"instruction": 0.7, "reasoning": 0.15, "span": 0.15}
    records = run(model, tasks, mix, tmp_path / "it", steps=12)
    assert any("length" in r for r in records)
    tuned = load_pretrained(tmp_path / "it" / "final").eval()
    assert tuned.config.length_head_trained
    result = generate(tuned, torch.tensor([[5, 6, 7]]))  # no canvas_len needed now
    assert 1 <= int(result.lengths[0]) <= tuned.config.max_response_len

    # Embedding stage from the base checkpoint.
    model = load_pretrained(tmp_path / "base" / "final")
    pairs = [([3 + i, 4 + i, 5 + i], [3 + i, 4 + i]) for i in range(30)]
    tasks, mix = embedding_stage(pairs, model.config, 8)
    records = run(model, tasks, mix, tmp_path / "embed", steps=10)
    assert {"fast", "reasoning"} <= set(records[-1])


def test_train_script_base_stage(tmp_path):
    import importlib.util
    from pathlib import Path

    toy_corpus(tmp_path / "corpus", count=50)
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main(["--stage", "base", "--preset", "tiny", "--corpus", str(tmp_path / "corpus"),
                 "--out", str(tmp_path / "run"), "--steps", "6", "--batch-size", "4"])
    assert (tmp_path / "run" / "final" / "config.json").exists()


def test_resume_matches_uninterrupted_run(tmp_path):
    """Stop at step 6, resume from the saved state: the result must equal a run without a stop."""
    config = tiny_config(moe=MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,),
                                       readout_layers=(2,)))
    corpus = toy_corpus(tmp_path / "corpus")

    def fresh_run():
        torch.manual_seed(0)
        model = NuriModel(config)
        tasks, mix = base_stage(corpus, config, batch_size=4, seed=3)
        settings = TrainSettings(out_dir=str(tmp_path / "unused"), total_steps=12, peak_lr=3e-3,
                                 warmup_steps=2, save_every=6, log_every=1, mix=mix)
        return model, tasks, settings

    model, tasks, settings = fresh_run()
    settings.out_dir = str(tmp_path / "full")
    train(model, tasks, settings, device=torch.device("cpu"))

    resumed, tasks, settings = fresh_run()
    settings.out_dir = str(tmp_path / "resumed")
    records = train(resumed, tasks, settings, device=torch.device("cpu"),
                    resume_from=tmp_path / "full" / "training_state" / "step-6.pt")

    assert records[0]["step"] == 6
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name]), name
    full_last = json.loads((tmp_path / "full" / "train_log.jsonl").read_text().splitlines()[-1])
    assert records[-1]["tokens"] == full_last["tokens"]


def test_resume_checks_settings_and_handles_finished_runs(tmp_path):
    import pytest

    config = tiny_config()
    corpus = toy_corpus(tmp_path / "corpus", count=50)

    def run(out, steps=6, save_every=3, resume=None, log_every=1):
        torch.manual_seed(0)
        model = NuriModel(config)
        tasks, mix = base_stage(corpus, config, batch_size=4)
        settings = TrainSettings(out_dir=str(tmp_path / out), total_steps=steps, peak_lr=1e-3, warmup_steps=1,
                                 save_every=save_every, log_every=log_every, mix=mix)
        return train(model, tasks, settings, device=torch.device("cpu"), resume_from=resume)

    run("first")
    state = tmp_path / "first" / "training_state"
    with pytest.raises(ValueError, match="total_steps"):
        run("longer", steps=8, resume=state / "step-3.pt")
    assert run("cadence", save_every=0, log_every=2, resume=state / "step-3.pt")  # output cadence may change
    assert run("done", resume=state / "final.pt") == []  # nothing left to do


def test_grad_accum_matches_one_big_batch(tmp_path):
    """Two micro-batches of the same task give the gradient of their mean loss."""
    from nuri.trainer import Task

    config = tiny_config()
    batches = [build_instruction_batch([([5, 6, 7], [8, 9]), ([10, 11], [12])], config,
                                       torch.Generator().manual_seed(i)) for i in range(2)]

    def run(accum, out):
        torch.manual_seed(0)
        model = NuriModel(config)
        source = iter(batches * 4)
        tasks = {"instruction": Task("instruction", lambda: next(source))}
        settings = TrainSettings(out_dir=str(tmp_path / out), total_steps=1, peak_lr=1e-2, warmup_steps=0,
                                 decay_fraction=0.0, mix={"instruction": 1.0}, grad_accum=accum, progress=False)
        record = train(model, tasks, settings, device=torch.device("cpu"))[0]
        return model, record

    accumulated, record = run(2, "accum")
    # Reference: average the two losses by hand and take one AdamW step.
    torch.manual_seed(0)
    reference = NuriModel(config)
    from nuri.trainer import canvas_loss, make_optimizer
    optimizer = make_optimizer(reference, TrainSettings(out_dir="unused", total_steps=1, peak_lr=1e-2))
    for group in optimizer.param_groups:
        group["lr"] = 1e-2
    # Replay the trainer's own random draws (seed 0): a task pick, then a depth, per micro-batch.
    rng = random.Random(0)
    losses = []
    for batch in batches:
        rng.choices(["instruction"], [1.0])
        losses.append(canvas_loss(reference, batch, rng.choice(config.train_depths), 0.1)["loss"])
    (sum(losses) / 2).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    optimizer.step()
    for name, tensor in reference.state_dict().items():
        assert torch.allclose(tensor, accumulated.state_dict()[name], atol=1e-6), name
    assert abs(record["loss"] - (sum(losses) / 2).item()) < 1e-5


def test_checkpointing_flag_does_not_change_training(tmp_path):
    corpus = toy_corpus(tmp_path / "corpus", count=30)

    def run(flag, out):
        torch.manual_seed(0)
        config = tiny_config()
        model = NuriModel(config)
        tasks, mix = base_stage(corpus, config, batch_size=4)
        settings = TrainSettings(out_dir=str(tmp_path / out), total_steps=3, peak_lr=1e-3, warmup_steps=1,
                                 mix=mix, gradient_checkpointing=flag, progress=False)
        train(model, tasks, settings, device=torch.device("cpu"))
        return model

    plain, saved = run(False, "plain"), run(True, "ckpt")
    for name, tensor in plain.state_dict().items():
        assert torch.allclose(tensor, saved.state_dict()[name], atol=1e-6), name
