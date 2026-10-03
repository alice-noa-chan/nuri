import json

import pytest
from tokenizers import Tokenizer

from tools.train_tokenizer import SPECIAL_TOKENS, read_records, train_variant


def test_byte_level_tokenizer_covers_unseen_unicode_and_keeps_model_ids(tmp_path):
    corpus = tmp_path / "input.txt"
    corpus.write_text("누리는 문장을 읽는다.\nA small model reads text.\n日本語の短い文章。\n", encoding="utf-8")
    out = tmp_path / "tokenizer"
    metrics = train_variant(corpus, out, 384)
    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    for text in ("새로운 한글🙂", "Unseen symbols: Ω×∑", "日本語と改行\n  空白"):
        encoding = tokenizer.encode(text)
        assert encoding.ids
        assert tokenizer.decode(encoding.ids) == text
    assert [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS[:3]] == [0, 1, 2]
    assert metrics["evaluation_split"] == "training_input"
    assert metrics["actual_vocab_size"] <= 384
    assert str(corpus) not in (out / "metrics.json").read_text(encoding="utf-8")


def test_compression_uses_held_out_language_groups(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    records = [{"text": "안녕하세요", "lang": "ko"}, {"text": "Hello there", "lang": "en"}]
    text = "\n".join(json.dumps(x, ensure_ascii=False) for x in records)
    train.write_text(text, encoding="utf-8")
    evaluation.write_text(text, encoding="utf-8")
    metrics = train_variant(train, tmp_path / "output", 320, file_format="jsonl", eval_file=evaluation)
    assert metrics["evaluation_split"] == "held_out"
    assert set(metrics["compression"]) == {"ko", "en"}
    assert metrics["compression"]["ko"]["bytes"] == len("안녕하세요".encode("utf-8"))


def test_empty_input_does_not_publish_a_tokenizer(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no nonempty"):
        train_variant(source, tmp_path / "out", 320)
    assert not (tmp_path / "out" / "tokenizer.json").exists()


def test_invalid_records_are_rejected_without_reprinting_text(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"text":17}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="string text"):
        list(read_records(source, "jsonl"))


def test_existing_artifacts_are_not_overwritten(tmp_path):
    source = tmp_path / "text.txt"
    source.write_text("Simple input\n", encoding="utf-8")
    out = tmp_path / "out"
    train_variant(source, out, 320)
    previous = (out / "tokenizer.json").read_bytes()
    with pytest.raises(FileExistsError):
        train_variant(source, out, 320)
    assert previous == (out / "tokenizer.json").read_bytes()
