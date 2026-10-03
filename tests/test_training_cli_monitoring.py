"""Exercise monitoring flags and preparation output through the real CLI."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.data import write_corpus
from scripts import train as train_cli


def prepare_inputs(tmp_path):
    words = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"]
    words += [f"word{i}" for i in range(32)]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    corpus_dir = tmp_path / "corpus"
    write_corpus(corpus_dir, [[8 + (row + pos) % 32 for pos in range(20)] for row in range(12)])
    return tokenizer_dir, corpus_dir


def scalar_steps(folder, tag="loss/total"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return [point.step for point in EventAccumulator(str(folder)).Reload().Scalars(tag)]


@pytest.mark.parametrize("custom_dir", [False, True])
def test_cli_enables_default_or_custom_events_and_disables_bars(tmp_path, capsys, custom_dir):
    pytest.importorskip("tensorboard")
    tokenizer, corpus = prepare_inputs(tmp_path)
    out = tmp_path / "base"
    event_dir = tmp_path / "custom-events" if custom_dir else out / "tensorboard"
    event_flags = ["--tensorboard-dir", str(event_dir)] if custom_dir else ["--tensorboard"]
    train_cli.main(["--stage", "base", "--preset", "tiny", "--tokenizer", str(tokenizer),
                    "--corpus", str(corpus), "--out", str(out), "--steps", "4", "--batch-size", "3",
                    "--log-every", "1", "--no-progress", *event_flags])
    output = capsys.readouterr()
    assert not output.err
    for phase in ("Loading tokenizer", "Initializing model", "Opening the token corpus",
                  "Preparing training tasks", "Starting training", "TensorBoard events", "done:"):
        assert phase in output.out
    settings = json.loads((out / "train_settings.json").read_text())
    assert settings["tensorboard_dir"] == str(event_dir)
    assert settings["stage"] == "base" and settings["progress"] is False
    assert scalar_steps(event_dir) == [1, 2, 3, 4]


def test_it_and_embed_show_preparation_progress_and_record_their_losses(tmp_path, capsys):
    pytest.importorskip("tensorboard")
    tokenizer, corpus = prepare_inputs(tmp_path)
    train_cli.main(["--stage", "base", "--preset", "tiny", "--tokenizer", str(tokenizer),
                    "--corpus", str(corpus), "--out", str(tmp_path / "base"), "--steps", "2",
                    "--batch-size", "3", "--no-progress"])
    capsys.readouterr()
    assert not (tmp_path / "base/tensorboard").exists()  # events are opt-in
    instructions = tmp_path / "instructions.jsonl"
    examples = [{"messages": [{"role": "user", "content": "word0 word1"}], "response": "word2 word3"}] * 4
    instructions.write_text("\n".join(json.dumps(example) for example in examples), encoding="utf-8")
    train_cli.main(["--stage", "it", "--init", str(tmp_path / "base/final"),
                    "--instructions", str(instructions), "--out", str(tmp_path / "it"),
                    "--steps", "3", "--batch-size", "3", "--reasoning-examples", "0",
                    "--tensorboard", "--log-every", "1"])
    output = capsys.readouterr()
    assert "Loading initial checkpoint" in output.out
    assert "it | tokenizing instructions" in output.err
    assert "it/instruction" in output.err and "length=" in output.err
    assert scalar_steps(tmp_path / "it/tensorboard", "loss/instruction/length") == [1, 2, 3]

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps({"query": f"word{i} word{i + 1}",
                                          "positive": f"word{i + 1} word{i}"}) for i in range(6)), encoding="utf-8")
    train_cli.main(["--stage", "embed", "--init", str(tmp_path / "base/final"), "--pairs", str(pairs),
                    "--out", str(tmp_path / "embed"), "--steps", "3", "--batch-size", "3",
                    "--tensorboard", "--log-every", "1"])
    output = capsys.readouterr()
    assert "embed | tokenizing pairs" in output.err
    assert "embed/pairs" in output.err and "fast=" in output.err and "reasoning=" in output.err
    assert scalar_steps(tmp_path / "embed/tensorboard", "loss/pairs/fast") == [1, 2, 3]
    assert scalar_steps(tmp_path / "embed/tensorboard", "loss/pairs/reasoning") == [1, 2, 3]


@pytest.mark.parametrize("cadence", ["0", "-3"])
def test_cli_rejects_nonpositive_metric_cadence(cadence, capsys):
    with pytest.raises(SystemExit) as error:
        train_cli.parse_args(["--stage", "base", "--out", "unused", "--steps", "10", "--log-every", cadence])
    assert error.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err
