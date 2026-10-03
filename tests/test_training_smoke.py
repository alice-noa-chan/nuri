"""End-to-end CPU smoke test: base -> instruction tuning -> embedding on toy data."""

import json
import random

import torch

from nuri.checkpoint import load_pretrained
from nuri.config import MoEConfig, tiny_config
from nuri.data import TokenCorpus, write_corpus
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
