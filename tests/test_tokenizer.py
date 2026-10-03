import importlib.util
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.data import write_corpus
from nuri.model import NuriModel
from nuri.tokenizer import check_tokenizer, load_tokenizer

WORDS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>", "[UNK]"] + \
        [f"w{i}" for i in range(52)]


def make_tokenizer(folder: Path, words=WORDS) -> Path:
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return folder


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_tokenizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_tokenizer_catches_size_and_special_id_mismatch(tmp_path):
    tokenizer = load_tokenizer(make_tokenizer(tmp_path / "ok"))
    check_tokenizer(tokenizer, tiny_config(vocab_size=len(WORDS)))
    with pytest.raises(ValueError, match="expects"):
        check_tokenizer(tokenizer, tiny_config(vocab_size=128))
    swapped = ["[MASK]", "[PAD]"] + WORDS[2:]
    with pytest.raises(ValueError, match="PAD|MASK"):
        check_tokenizer(load_tokenizer(make_tokenizer(tmp_path / "bad", swapped)), tiny_config(vocab_size=len(WORDS)))


def test_base_stage_uses_tokenizer_size_and_ships_tokenizer(tmp_path):
    tok_dir = make_tokenizer(tmp_path / "tok")
    write_corpus(tmp_path / "corpus", [[8 + (i + j) % 50 for j in range(30)] for i in range(40)])
    script = load_script()
    script.main(["--stage", "base", "--preset", "tiny", "--tokenizer", str(tok_dir), "--corpus",
                 str(tmp_path / "corpus"), "--out", str(tmp_path / "base"), "--steps", "4", "--batch-size", "4"])
    final = tmp_path / "base" / "final"
    assert json.loads((final / "config.json").read_text())["vocab_size"] == len(WORDS)
    assert (final / "tokenizer.json").exists() and (final / "tokenizer_config.json").exists()

    # The embed stage finds the tokenizer inside --init on its own.
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(json.dumps({"query": f"w{i} w{i + 1}", "positive": f"w{i + 1} w{i}"}) for i in range(8)),
        encoding="utf-8")
    script.main(["--stage", "embed", "--init", str(final), "--pairs", str(tmp_path / "pairs.jsonl"),
                 "--out", str(tmp_path / "embed"), "--steps", "3", "--batch-size", "4"])
    assert (tmp_path / "embed" / "final" / "tokenizer.json").exists()


def test_it_stage_rejects_a_mismatched_tokenizer(tmp_path):
    model = NuriModel(tiny_config())  # vocab 128
    save_pretrained(model, tmp_path / "init")
    with pytest.raises(ValueError):
        load_script().main(["--stage", "it", "--init", str(tmp_path / "init"), "--tokenizer",
                            str(make_tokenizer(tmp_path / "tok")), "--instructions", "unused",
                            "--out", str(tmp_path / "it"), "--steps", "1"])



def test_embed_resume_needs_no_init(tmp_path):
    tok_dir = make_tokenizer(tmp_path / "tok")
    write_corpus(tmp_path / "corpus", [[8 + (i + j) % 50 for j in range(30)] for i in range(40)])
    script = load_script()
    script.main(["--stage", "base", "--preset", "tiny", "--tokenizer", str(tok_dir), "--corpus",
                 str(tmp_path / "corpus"), "--out", str(tmp_path / "base"), "--steps", "2", "--batch-size", "4"])
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(json.dumps({"query": f"w{i} w{i + 1}", "positive": f"w{i + 1} w{i}"}) for i in range(8)),
        encoding="utf-8")
    common = ["--stage", "embed", "--pairs", str(tmp_path / "pairs.jsonl"), "--out", str(tmp_path / "embed"),
              "--steps", "4", "--batch-size", "4", "--save-every", "2"]
    script.main(common + ["--init", str(tmp_path / "base" / "final")])
    # Resume without --init: the model comes from the saved state, the tokenizer from --tokenizer.
    script.main(common + ["--tokenizer", str(tok_dir), "--resume",
                          str(tmp_path / "embed" / "training_state" / "step-2.pt")])
