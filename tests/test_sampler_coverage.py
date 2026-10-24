"""Retained candidates/tails must survive packing, bucketing and resume."""
import copy
import random

import pytest
import torch

from nuri.config import tiny_config
from nuri.data import BufferedShuffledPasses, TokenCorpus, write_corpus
from nuri.pipeline import base_stage
from nuri.trainer import _restore_task_random, _task_random_state


def config():
    return tiny_config(vocab_size=512, max_input_len=64, max_response_len=16)


def test_bucket_candidates_are_all_used_before_the_pool_is_replaced(tmp_path):
    write_corpus(tmp_path, [[3 + i] * 64 for i in range(256)])
    tasks, _ = base_stage(TokenCorpus(tmp_path), config(), 8, bucket=True,
                          shuffled_passes=True)
    sampler = tasks["span"].sampler
    seen = []
    for _ in range(256):
        batch = tasks["span"].next_batch()
        seen.extend(batch["targets"][:, 0].tolist())
        assert len(sampler.span_candidates) <= 64
        assert len(seen) == len(set(seen))
        if sampler.passes == [1] and sampler.positions == [0] and not sampler.span_candidates:
            break
    assert sorted(seen) == list(range(3, 259))
    assert sampler.passes == [1] and not sampler.span_candidates


def test_packed_tail_is_used_before_another_source_piece(tmp_path):
    write_corpus(tmp_path, [[3] * 48, [4] * 64])
    tasks, _ = base_stage(TokenCorpus(tmp_path), config(), 1, seed=0,
                          pack_mlm=True, shuffled_passes=True)
    first = tasks["mlm"].next_batch()
    sampler = tasks["mlm"].sampler
    assert first["labels"][0].tolist() == [3] * 48 + [4] * 16
    assert sampler.mlm_tail == [4] * 48
    second = tasks["mlm"].next_batch()
    assert second["labels"][0, :48].tolist() == [4] * 48
    assert second["segments"][0, :48].eq(1).all()
    assert second["segments"][0, 48:].eq(2).all()


def test_both_buffers_survive_serialization_and_interleaved_resume(tmp_path):
    path = tmp_path / "corpus"
    write_corpus(path, [[3 + i] * (20 + i % 45) for i in range(200)])

    def fresh():
        return base_stage(TokenCorpus(path), config(), 3, seed=19,
                          pack_mlm=True, bucket=True, shuffled_passes=True)[0]

    tasks = fresh()
    for _ in range(30):
        tasks["span"].next_batch()
        tasks["mlm"].next_batch()
        sampler = tasks["span"].sampler
        if sampler.span_candidates and sampler.mlm_tail:
            break
    assert sampler.span_candidates and sampler.mlm_tail
    snapshot = _task_random_state(tasks)
    frozen = copy.deepcopy(snapshot["span"]["sampler"])
    state_path = tmp_path / "state.pt"
    torch.save(snapshot, state_path)
    names = ["span", "mlm", "mlm", "span"] * 15
    expected = [tasks[name].next_batch() for name in names]
    assert snapshot["span"]["sampler"] == frozen  # snapshots do not alias live queues
    resumed = fresh()
    _restore_task_random(resumed, torch.load(state_path, weights_only=False))
    for name, reference in zip(names, expected):
        actual = resumed[name].next_batch()
        assert actual.keys() == reference.keys()
        for key in actual:
            if torch.is_tensor(actual[key]):
                assert torch.equal(actual[key], reference[key]), key
            else:
                assert actual[key] == reference[key]


def test_legacy_and_oversized_buffer_states_are_rejected(tmp_path):
    write_corpus(tmp_path, [[3] * 32, [4] * 32])
    sampler = BufferedShuffledPasses(TokenCorpus(tmp_path), 64, 0, 8, 512)
    with pytest.raises(ValueError, match="saved buffers"):
        sampler.load_state_dict({"passes": [0], "positions": [0]})
    for update in ({"span_candidates": [[3, 4]] * 9}, {"mlm_tail": [3] * 65},
                   {"mlm_tail": [512]}, {"span_candidates": [[3]]}):
        state = sampler.state_dict()
        state.update(update)
        with pytest.raises(ValueError, match="invalid|two tokens"):
            sampler.load_state_dict(state)
    assert sampler.positions == [0] and sampler.mlm_tail == []


def test_special_only_tail_windows_do_not_create_targetless_rows(tmp_path):
    write_corpus(tmp_path, [[3] + [2] * 63, [4] + [2] * 63])
    sampler = BufferedShuffledPasses(TokenCorpus(tmp_path), 64, 0, 8, 512)
    usable = lambda seq: any(t not in (0, 1, 2) for t in seq)
    rng = random.Random(0)
    assert sampler.packed_piece(16, rng, usable)[0] == 3
    assert sampler.packed_piece(16, rng, usable)[0] == 4
    with pytest.raises(ValueError, match="positive"):
        sampler.packed_piece(0, rng, usable)
