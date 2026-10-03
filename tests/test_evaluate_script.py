import importlib.util
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from nuri.checkpoint import save_pretrained
from nuri.config import tiny_config
from nuri.data import write_corpus
from nuri.model import NuriModel
from nuri.tasks import reasoning_examples

SPECIALS = ["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>"]


def train_small_tokenizer(folder: Path) -> Tokenizer:
    """A tiny byte-level BPE trained on the task texts themselves (Korean and English)."""
    texts = [m[0]["content"] + " " + a for m, a in reasoning_examples(200, seed=1, split="eval")]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer)
    folder.mkdir(parents=True)
    tokenizer.save(str(folder / "tokenizer.json"))
    return tokenizer


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_suite_runs_on_a_tiny_checkpoint(tmp_path):
    tokenizer = train_small_tokenizer(tmp_path / "tok")
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    config = tiny_config(vocab_size=size, max_input_len=128, max_response_len=8, length_head_trained=True)
    save_pretrained(NuriModel(config), tmp_path / "ckpt", tokenizer_dir=tmp_path / "tok")
    write_corpus(tmp_path / "corpus", [[10 + (i * 7 + j) % 300 for j in range(40)] for i in range(30)])
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(json.dumps({"query": f"question {i}", "positive": f"answer {i}"}) for i in range(10)),
        encoding="utf-8")

    report = load_script().main([
        "--checkpoint", str(tmp_path / "ckpt"), "--suites", *("reasoning", "curves", "interventions", "mlm",
                                                               "denoise", "retrieval", "diagnostics"),
        "--corpus", str(tmp_path / "corpus"), "--pairs", str(tmp_path / "pairs.jsonl"),
        "--examples", "6", "--documents", "8", "--output", str(tmp_path / "report.json")])

    assert set(report["reasoning"]) == {"shared", "eval"}
    assert set(report["reasoning"]["shared"]) == {"oracle_length", "predicted_length"}
    assert [r["steps"] for r in report["depth_curve"]] == [1, 2, 4, 8, 16]
    assert [r["rounds"] for r in report["rounds_curve"]] == [1, 2, 4, 8]
    assert set(report["interventions"]) == {"normal", "zero_r", "shuffled_r"}
    assert report["mlm"]["targets"] > 0 and len(report["diagnostics"]) == 17
    assert [(r["steps"], r["mask_ratio"]) for r in report["denoise"]][:3] == [(1, 0.15), (1, 0.5), (1, 1.0)]
    # tiny embed_dim is 32, so only the full-size embedding is evaluated (64/128 do not fit)
    assert {(r["mode"], r["dim"]) for r in report["retrieval"]} == {("fast", None), ("reasoning", None)}
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report


def test_base_checkpoint_uses_oracle_length_only(tmp_path):
    tokenizer = train_small_tokenizer(tmp_path / "tok")
    config = tiny_config(vocab_size=tokenizer.get_vocab_size(with_added_tokens=True), max_input_len=128)
    save_pretrained(NuriModel(config), tmp_path / "ckpt", tokenizer_dir=tmp_path / "tok")
    report = load_script().main(["--checkpoint", str(tmp_path / "ckpt"), "--suites", "reasoning",
                                 "--examples", "4"])
    assert report["oracle_length_only"] and set(report["reasoning"]["shared"]) == {"oracle_length"}
