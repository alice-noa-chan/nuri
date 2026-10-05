"""A Base initializer must preserve trained weights instead of starting from scratch."""

import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.data import write_corpus
from nuri.model import NuriModel
from scripts import train as train_cli


def test_base_init_loads_weights_and_tokenizer_before_fresh_training(tmp_path, monkeypatch):
    words = ["[PAD]", "[MASK]", "[EOS]", "[UNK]"] + [f"word{i}" for i in range(28)]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    config = tiny_config(vocab_size=len(words), hidden_size=32, num_heads=4, ffn_size=48,
                         length_head_trained=True)
    torch.manual_seed(73)
    original = NuriModel(config)
    checkpoint = tmp_path / "initializer"
    save_pretrained(original, checkpoint, dtype=torch.float32, tokenizer_dir=tokenizer_dir)
    corpus = tmp_path / "corpus"
    write_corpus(corpus, [[4 + (row + i) % 28 for i in range(24)] for row in range(8)])
    original_train = train_cli.train
    inspected = []

    def inspect_then_train(model, tasks, settings, resume_from):
        assert model.config.hidden_size == 32  # the tiny preset would build 64
        assert not model.config.length_head_trained
        assert settings.tokenizer_dir == str(checkpoint)
        assert resume_from is None
        for name, tensor in original.state_dict().items():
            assert torch.equal(tensor, model.state_dict()[name]), name
        inspected.append(True)
        return original_train(model, tasks, settings, device=torch.device("cpu"))

    monkeypatch.setattr(train_cli, "train", inspect_then_train)
    out = tmp_path / "continued"
    train_cli.main(["--stage", "base", "--preset", "tiny", "--init", str(checkpoint),
                    "--corpus", str(corpus), "--out", str(out), "--steps", "1",
                    "--batch-size", "2", "--seed", "19", "--no-progress"])
    assert inspected == [True]
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [0]
    state = torch.load(out / "training_state/final.pt", weights_only=False)
    assert state["step"] == 1
    assert all(int(row["step"]) == 1 for row in state["optimizer"]["state"].values())


def test_resume_has_priority_over_initializer(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    write_corpus(corpus, [[3, 4, 5, 6] for _ in range(4)])
    resumed = tiny_config()
    monkeypatch.setattr(train_cli, "training_state_config", lambda path: resumed)

    def refuse_initializer(*args, **kwargs):
        raise AssertionError("A resumed run must not load initializer weights.")

    monkeypatch.setattr(train_cli, "load_pretrained", refuse_initializer)
    calls = []

    def inspect_resume(model, tasks, settings, resume_from):
        assert model.config is resumed
        calls.append(resume_from)
        return []

    monkeypatch.setattr(train_cli, "train", inspect_resume)
    train_cli.main(["--stage", "base", "--init", str(tmp_path / "unused"),
                    "--resume", str(tmp_path / "state.pt"), "--corpus", str(corpus),
                    "--out", str(tmp_path / "resume"), "--steps", "3", "--no-progress"])
    assert calls == [str(tmp_path / "state.pt")]
