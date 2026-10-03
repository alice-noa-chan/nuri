import random

import torch

from nuri.config import tiny_config
from nuri.data import (TokenCorpus, build_instruction_batch, build_mlm_batch, build_pair_batch,
                       build_span_batch, chat_prompt_text, write_corpus)

CONFIG = tiny_config()


def test_corpus_roundtrip(tmp_path):
    docs = [[5, 6, 7], [], [8], list(range(10, 50))]
    assert write_corpus(tmp_path, docs) == 3  # empty documents are skipped
    corpus = TokenCorpus(tmp_path)
    assert len(corpus) == 3
    assert [corpus.document(i) for i in range(3)] == [[5, 6, 7], [8], list(range(10, 50))]
    assert len(corpus.sample(5, random.Random(0))) == 5


def test_mlm_batch_windows_fit_input_limit():
    docs = [list(range(3, 100)), [3, 4, 5]]
    batch = build_mlm_batch(docs, CONFIG, random.Random(0), torch.Generator().manual_seed(0))
    assert batch["input_ids"].shape == (2, CONFIG.max_input_len)
    assert batch["target_mask"].any(dim=1).all()
    assert not batch["target_mask"][~batch["input_valid"]].any()


def test_span_batch_keeps_answer_out_of_input():
    docs = [list(range(3, 80)) for _ in range(4)]
    batch = build_span_batch(docs, CONFIG, random.Random(0), torch.Generator().manual_seed(0), min_span=4)
    for row in range(4):
        source = batch["input_ids"][row][batch["input_valid"][row]].tolist()
        target = batch["targets"][row][batch["canvas_valid"][row]].tolist()
        assert not set(source) & set(target)  # tokens are unique, so no overlap means no leak
        assert (batch["canvas"][row][batch["loss_mask"][row]] == CONFIG.mask_id).all()


def test_instruction_batch_appends_eos_and_drops_long_answers():
    examples = [([5, 6], [7, 8, 9]), ([5], list(range(3, 40))), ([], [7])]
    batch = build_instruction_batch(examples, CONFIG, torch.Generator().manual_seed(0))
    assert batch["dropped"] == 2
    assert batch["targets"][0].tolist() == [7, 8, 9, CONFIG.eos_id]
    assert batch["lengths"].tolist() == [4]


def test_pair_batch_and_chat_prompt():
    batch = build_pair_batch([([5, 6], [7]), ([8], [9, 10, 11])], CONFIG)
    assert batch["query_ids"].shape == (2, 2) and batch["doc_ids"].shape == (2, 3)
    text = chat_prompt_text([{"role": "user", "content": "hi"}])
    assert text == "<|user|>hi[EOS]<|assistant|>"
