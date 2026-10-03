"""Exercise text inference with a real local tokenizer and saved tied weights."""

import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from nuri.checkpoint import load_pretrained, save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel
from tools.infer_nuri import infer, load_tokenizer, main, tokenize_batch


@pytest.fixture
def checkpoint(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello world", "한국어 문장", "日本語", "Test input"],
        trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=["[PAD]", "[MASK]", "[EOS]", "[SEP]", "<|system|>", "<|user|>", "<|assistant|>"]),
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    model = NuriModel(tiny_config(vocab_size=tokenizer.get_vocab_size(), max_input_len=64)).eval()
    save_pretrained(model, tmp_path, dtype=torch.float32)
    loaded = load_pretrained(tmp_path).eval()
    return tmp_path, loaded, load_tokenizer(tmp_path, loaded)


def test_embedding_text_batch_matches_individual_vectors(checkpoint):
    _, model, tokenizer = checkpoint
    texts = ["Hello", "한국어 문장"]
    batched = infer(model, tokenizer, texts, task="embed", mode="reasoning", dim=16)
    vectors = torch.tensor(batched["embeddings"])
    assert vectors.shape == (2, 16)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(2), atol=1e-6)
    for index, text in enumerate(texts):
        individual = infer(model, tokenizer, [text], task="embed", mode="reasoning", dim=16)
        assert torch.allclose(vectors[index], torch.tensor(individual["embeddings"])[0], atol=2e-6)


def test_base_generation_requires_length_and_disallows_mask_or_pad(checkpoint):
    _, model, tokenizer = checkpoint
    with pytest.raises(ValueError, match="canvas_len"):
        infer(model, tokenizer, ["Hello"], task="generate")
    output = infer(model, tokenizer, ["Hello", "日本語"], canvas_len=5, rounds=2)
    assert output["canvas_lengths"] == [5, 5]
    assert len(output["answers"]) == 2
    assert all(token not in {model.config.pad_id, model.config.mask_id} for row in output["token_ids"] for token in row)


def test_chat_prompt_excludes_answer_and_ends_with_assistant_tag(checkpoint):
    _, model, tokenizer = checkpoint
    ids, valid = tokenize_batch(tokenizer, model, ["Hello"], chat=True)
    rendered = tokenizer.decode(ids[0, valid[0]].tolist(), skip_special_tokens=False)
    assert rendered == "<|user|>Hello[EOS]<|assistant|>"


def test_long_inputs_require_explicit_truncation(checkpoint):
    _, model, tokenizer = checkpoint
    with pytest.raises(ValueError, match="max_input_len"):
        tokenize_batch(tokenizer, model, ["abcdef " * 100])
    ids, _ = tokenize_batch(tokenizer, model, ["abcdef " * 100], truncate=True)
    assert ids.shape[1] == model.config.max_input_len


def test_invalid_inputs_and_tokenizer_mismatch_fail_clearly(checkpoint):
    folder, model, tokenizer = checkpoint
    for arguments in ({"texts": [""]}, {"texts": ["Hello"], "rounds": 0},
                      {"texts": ["Hello"], "task": "embed", "dim": 33}):
        with pytest.raises(ValueError):
            infer(model, tokenizer, **arguments)
    wrong = NuriModel(tiny_config(vocab_size=tokenizer.get_vocab_size() + 1))
    with pytest.raises(ValueError, match="vocabulary"):
        load_tokenizer(folder, wrong)


def test_cli_loads_checkpoint_and_writes_json(checkpoint, capsys):
    folder, _, _ = checkpoint
    output_path = folder / "embeddings.json"
    assert main(["--checkpoint", str(folder), "--task", "embed", "--text", "Hello", "--dim", "16",
                 "--output", str(output_path)]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["dimension"] == 16
    assert len(result["embeddings"][0]) == 16
    assert '"task": "embed"' in capsys.readouterr().out
