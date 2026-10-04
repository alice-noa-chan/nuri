import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import frequency_baseline as fb  # noqa: E402

from nuri import evaluation  # noqa: E402
from nuri.config import tiny_config  # noqa: E402
from nuri.data import TokenCorpus, write_corpus  # noqa: E402
from nuri.model import NuriModel  # noqa: E402


def corpus(folder, count=80, seed=0):
    rng = random.Random(seed)
    # A skewed vocabulary, so the frequency line differs clearly from uniform.
    docs = [[rng.choice([5, 5, 5, 6, 6, 7, 8, 9, 10 + rng.randrange(50)]) for _ in range(rng.randint(5, 60))]
            for _ in range(count)]
    write_corpus(folder, docs)
    return TokenCorpus(folder)


def test_targets_are_exactly_those_of_the_mlm_suite(tmp_path):
    config = tiny_config()
    held_out = corpus(tmp_path / "eval")
    targets = fb.evaluation_targets(held_out, config, documents=40, seed=3)

    # The real suite on the same documents and seed reports the same target count.
    documents = held_out.sample(40, random.Random(3), usable=lambda d: len(d) >= 2)
    report = evaluation.mlm_accuracy(NuriModel(config), documents, seed=3)
    assert report["targets"] == len(targets)


def test_frequency_loss_matches_a_direct_computation():
    counts = np.array([0, 0, 10, 30, 60], dtype=np.int64)
    targets = torch.tensor([2, 3, 4, 4])
    p = (counts + 1) / (counts.sum() + len(counts))
    expected = float(-np.log(p[[2, 3, 4, 4]]).mean())
    assert fb.frequency_loss(targets, counts) == pytest.approx(expected)


def test_chunked_counts_equal_a_plain_bincount(tmp_path, monkeypatch):
    held_out = corpus(tmp_path / "eval")
    monkeypatch.setattr(fb, "CHUNK", 7)  # many small chunks
    counts = fb.token_counts([tmp_path / "eval"], 128)
    assert np.array_equal(counts, np.bincount(np.asarray(held_out.tokens, dtype=np.int64), minlength=128))


def test_cli_reports_lines_below_uniform(tmp_path):
    corpus(tmp_path / "eval", seed=1)
    corpus(tmp_path / "train", seed=2)
    config = tmp_path / "config.json"
    tiny_config().save(config)
    result = fb.main(["--corpus", str(tmp_path / "eval"), "--config", str(config), "--documents", "30",
                      "--fit-corpus", str(tmp_path / "train"), "--output", str(tmp_path / "line.json")])
    assert result["targets"] > 0
    assert result["eval_fitted"] <= result["fit"] + 0.5 and result["fit"] < result["uniform"] == math.log(128)
    assert json.loads((tmp_path / "line.json").read_text(encoding="utf-8"))["targets"] == result["targets"]
