"""Reject malformed token boundaries before sampling or allocating a model."""

import random

import numpy as np
import pytest

from nuri.data import MixedCorpus, TokenCorpus, write_corpus


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([0, 3], dtype=np.int32),
        np.array([[0, 3]], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([0, 4], dtype=np.int64),
        np.array([0, 0, 3], dtype=np.int64),
        np.array([0, 2, 1, 3], dtype=np.int64),
    ],
)
def test_invalid_corpus_offsets_are_rejected(tmp_path, offsets):
    np.array([4, 5, 6], dtype=np.uint32).tofile(tmp_path / "tokens.bin")
    np.save(tmp_path / "offsets.npy", offsets)
    with pytest.raises(ValueError, match="offsets"):
        TokenCorpus(tmp_path)


def test_partial_token_file_is_rejected(tmp_path):
    (tmp_path / "tokens.bin").write_bytes(b"abc")
    np.save(tmp_path / "offsets.npy", np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="uint32"):
        TokenCorpus(tmp_path)


def test_empty_corpus_written_by_writer_can_be_inspected(tmp_path):
    assert write_corpus(tmp_path, []) == 0
    corpus = TokenCorpus(tmp_path)
    assert len(corpus) == 0 and corpus.tokens.size == 0
    with pytest.raises(ValueError, match="at least one document"):
        MixedCorpus([(corpus, 1)])


@pytest.mark.parametrize("index", [-1, 1, True, 0.5])
def test_document_indices_are_checked(tmp_path, index):
    write_corpus(tmp_path, [[5, 6]])
    with pytest.raises(IndexError, match="document index"):
        TokenCorpus(tmp_path).document(index)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1, 0, True, "1"])
def test_invalid_mixture_weights_fail_before_sampling(tmp_path, weight):
    write_corpus(tmp_path, [[5, 6]])
    with pytest.raises(ValueError, match="weights"):
        MixedCorpus([(TokenCorpus(tmp_path), weight)])


def test_overflowing_mixture_total_is_rejected(tmp_path):
    write_corpus(tmp_path, [[5, 6]])
    corpus = TokenCorpus(tmp_path)
    with pytest.raises(ValueError, match="total"):
        MixedCorpus([(corpus, 1e308), (corpus, 1e308)])


def test_valid_corpus_keeps_sampling_reproducible(tmp_path):
    write_corpus(tmp_path, [[5], [6, 7], [8, 9, 10]])
    corpus = TokenCorpus(tmp_path)
    rng = random.Random(2)
    expected = random.Random(2)
    for _ in range(20):
        assert corpus.random_document(rng) == corpus.document(expected.randrange(3))


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_both_token_widths_read_the_same_documents(tmp_path, dtype):
    documents = [[5], [6, 65535], [8, 9, 10]]
    write_corpus(tmp_path, documents, dtype=dtype)
    corpus = TokenCorpus(tmp_path)
    assert corpus.tokens.dtype == dtype
    assert (tmp_path / "tokens.bin").stat().st_size == 6 * np.dtype(dtype).itemsize
    assert [corpus.document(i) for i in range(3)] == documents


def test_writer_defaults_to_uint16_and_refuses_ids_that_do_not_fit(tmp_path):
    write_corpus(tmp_path / "small", [[5, 6]])
    assert TokenCorpus(tmp_path / "small").tokens.dtype == np.uint16
    with pytest.raises(ValueError, match="0..65535"):
        write_corpus(tmp_path / "big", [[5, 65536]])
    with pytest.raises(ValueError, match="0..65535"):
        write_corpus(tmp_path / "negative", [[-1]])
    with pytest.raises(ValueError, match="uint16 or uint32"):
        write_corpus(tmp_path / "wide", [[5]], dtype=np.int64)
