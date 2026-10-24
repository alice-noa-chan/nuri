"""ShuffledPasses: every corpus piece once per pass, resumable, wired into the base stage."""

import random

import numpy as np
import pytest

from nuri.config import tiny_config
from nuri.data import MixedCorpus, ShuffledPasses, TokenCorpus, write_corpus
from nuri.pipeline import base_stage


def corpus(tmp_path, name, lengths, first_id=8):
    documents, next_id = [], first_id
    for length in lengths:
        documents.append([next_id + i for i in range(length)])
        next_id += length
    write_corpus(tmp_path / name, documents)
    return TokenCorpus(tmp_path / name)


def test_one_pass_returns_every_piece_exactly_once(tmp_path):
    source = corpus(tmp_path, "a", [3, 10, 4, 7])  # piece_len 4: 1 + 3 + 1 + 2 pieces
    sampler = ShuffledPasses(source, piece_len=4, seed=0)
    assert len(sampler) == 7
    rng = random.Random(0)
    pieces = [sampler.random_document(rng) for _ in range(7)]
    assert all(1 <= len(piece) <= 4 for piece in pieces)
    tokens = sorted(t for piece in pieces for t in piece)
    assert tokens == sorted(np.asarray(source.tokens, dtype=np.int64).tolist())
    assert sampler.state_dict() == {"passes": [1], "positions": [0]}
    # The next pass has a new order but again every piece.
    second = [sampler.random_document(rng) for _ in range(7)]
    assert sorted(map(tuple, second)) == sorted(map(tuple, pieces))
    assert second != pieces


def test_pieces_never_cross_documents(tmp_path):
    source = corpus(tmp_path, "a", [5, 6])
    sampler = ShuffledPasses(source, piece_len=4, seed=3)
    pieces = [sampler.random_document(random.Random(0)) for _ in range(4)]
    assert sorted(map(tuple, pieces)) == [
        (8, 9, 10, 11),
        (12,),
        (13, 14, 15, 16),
        (17, 18),
    ]


def test_state_dict_resumes_the_same_sequence(tmp_path):
    a = corpus(tmp_path, "a", [5] * 20)
    b = corpus(tmp_path, "b", [3] * 9, first_id=500)
    mixed = MixedCorpus([(a, 2.0), (b, 1.0)])
    reference, rng = ShuffledPasses(mixed, 2, seed=7), random.Random(1)
    for _ in range(25):
        reference.random_document(rng)
    state, rng_state = reference.state_dict(), rng.getstate()
    expected = [reference.random_document(rng) for _ in range(60)]

    resumed, rng = ShuffledPasses(mixed, 2, seed=7), random.Random()
    resumed.load_state_dict(state)
    rng.setstate(rng_state)
    assert [resumed.random_document(rng) for _ in range(60)] == expected
    with pytest.raises(ValueError, match="different number"):
        ShuffledPasses(a, 2, seed=7).load_state_dict(state)


def test_mixed_sources_keep_their_weights_and_cover_each_source(tmp_path):
    a = corpus(tmp_path, "a", [2] * 300)
    b = corpus(tmp_path, "b", [2] * 100, first_id=5000)
    sampler = ShuffledPasses(MixedCorpus([(a, 3.0), (b, 1.0)]), 2, seed=0)
    rng = random.Random(0)
    drawn = [tuple(sampler.random_document(rng)) for _ in range(380)]
    from_b = [piece for piece in drawn if piece[0] >= 5000]
    assert 70 <= len(from_b) <= 120  # about a quarter of the draws
    assert len(set(from_b)) == min(len(from_b), 100)  # no repeat inside a pass


def test_base_stage_uses_one_shared_pass_and_records_it(tmp_path):
    source = corpus(tmp_path, "a", [12] * 30)
    config = tiny_config()
    tasks, _ = base_stage(
        source, config, 4, seed=0, pack_mlm=True, shuffled_passes=True
    )
    assert tasks["mlm"].sampler is tasks["span"].sampler
    assert tasks["mlm"].options["sampling"] == {
        "kind": "shuffled_passes",
        "version": 2,
        "piece_len": config.max_input_len,
        "pieces": len(tasks["mlm"].sampler),
        "candidate_capacity": 4 * 8,
    }
    tasks["mlm"].next_batch()
    tasks["span"].next_batch()
    assert sum(tasks["mlm"].sampler.state_dict()["positions"]) > 0
    plain, _ = base_stage(source, config, 4, seed=0)
    assert plain["mlm"].sampler is None and "sampling" not in plain["mlm"].options


def test_interrupted_training_resumes_the_same_pass_exactly(tmp_path):
    import torch

    from nuri.model import NuriModel
    from nuri.trainer import TrainSettings, TrainingInterrupted, train
    from test_training_boundary import assert_same

    folder = tmp_path / "corpus"
    write_corpus(
        folder, [[3 + (row + p) % 60 for p in range(5 + row % 9)] for row in range(15)]
    )
    config = tiny_config(hidden_size=16, num_heads=2, ffn_size=32, input_layers=1,
                         readout_layers=1, embed_dim=8)  # fmt: skip

    def run(name, boundary=None, resume=None):
        torch.manual_seed(17)
        model = NuriModel(config)
        tasks, mix = base_stage(TokenCorpus(folder), config, 3, seed=9, pack_mlm=True,
                                bucket=True, shuffled_passes=True)  # fmt: skip
        settings = TrainSettings(str(tmp_path / name), 8, warmup_steps=2, mix=mix,
                                 progress=False, embed_dims=(4, 8),
                                 stop_after_completed_steps=boundary)  # fmt: skip
        train(model, tasks, settings, torch.device("cpu"), resume_from=resume)
        return tasks

    tasks = run("plain")
    assert tasks["mlm"].sampler.passes[0] >= 1  # the run went past one whole pass
    reference = torch.load(
        tmp_path / "plain/training_state/final.pt", weights_only=False
    )
    with pytest.raises(TrainingInterrupted) as stopped:
        run("gated", boundary=4)
    run("gated", resume=stopped.value.path)
    final = torch.load(tmp_path / "gated/training_state/final.pt", weights_only=False)
    for key in ("model", "optimizer", "step", "extra"):
        assert_same(reference[key], final[key])


def test_cli_allows_shuffled_passes_only_for_base():
    from scripts import train as cli

    with pytest.raises(SystemExit) as error:
        cli.parse_args(
            ["--stage", "it", "--out", "x", "--steps", "1", "--shuffled-passes"]
        )
    assert error.value.code == 2
    assert cli.parse_args(["--stage", "base", "--out", "x", "--steps", "1",
                           "--shuffled-passes"]).shuffled_passes  # fmt: skip
