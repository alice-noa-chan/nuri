import importlib.util
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from nuri.data import TokenCorpus


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "tokenize_corpus.py"
    spec = importlib.util.spec_from_file_location("tokenize_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenize_text_and_jsonl(tmp_path):
    vocab = {"[PAD]": 0, "[MASK]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4, "world": 5, "again": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "a.txt").write_text("hello world\nhello\n\nworld again hello\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(json.dumps({"text": "again world"}) + "\n", encoding="utf-8")

    load_script().main([str(tmp_path / "a.txt"), str(tmp_path / "b.jsonl"), "--tokenizer", str(tmp_path),
                        "--out", str(tmp_path / "corpus"), "--batch-size", "2"])
    corpus = TokenCorpus(tmp_path / "corpus")
    assert [corpus.document(i) for i in range(len(corpus))] == [[4, 5], [5, 6, 4], [6, 5]]
    stats = json.loads((tmp_path / "corpus" / "stats.json").read_text())
    assert stats == {"documents_seen": 4, "tokens": 7, "documents_written": 3}
