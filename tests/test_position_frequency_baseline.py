import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import position_frequency_baseline as pfb  # noqa: E402

from nuri.config import tiny_config  # noqa: E402
from nuri.data import TokenCorpus, write_corpus  # noqa: E402


def corpus(folder, count=60, seed=0):
    rng = random.Random(seed)
    docs = [[rng.choice([5, 5, 6, 7, 10 + rng.randrange(40)]) for _ in range(rng.randint(20, 60))]
            for _ in range(count)]
    write_corpus(folder, docs)
    return TokenCorpus(folder)


def test_counts_cover_every_sampled_span_token(tmp_path):
    config = tiny_config()
    counts = pfb.span_position_counts(corpus(tmp_path / "train"), config, samples=50, seed=1)
    assert counts.shape == (config.max_response_len, config.vocab_size)
    # Every span has at least min(MIN_SPAN, remaining) >= 1 tokens, so position 0 is counted once per sample.
    assert counts[0].sum() == 50
    # Rows are non-increasing in total: a span that reaches position i also covers every earlier position.
    totals = counts.sum(axis=1)
    assert np.all(totals[:-1] >= totals[1:])


def test_positional_distribution_follows_position_counts():
    # Position 0 always holds token 2, position 1 always token 3.
    counts = np.zeros((2, 5), dtype=np.int64)
    counts[0, 2] = 100
    counts[1, 3] = 100
    pooled, positional = pfb.log_probabilities(counts, prior_strength=1.0)
    assert np.allclose(np.exp(pooled).sum(), 1.0) and np.allclose(np.exp(positional).sum(axis=1), 1.0)
    result = pfb.score([[2, 3]], pooled, positional)
    assert result["positional"]["correct"] == 2
    assert result["pooled"]["correct"] == 1  # one argmax token cannot fit both positions
    assert result["positional"]["row_mean_ce"] < result["pooled"]["row_mean_ce"]


def test_row_mean_differs_from_token_mean():
    counts = np.ones((4, 3), dtype=np.int64)
    pooled, positional = pfb.log_probabilities(counts, prior_strength=1.0)
    pooled = np.log(np.array([0.5, 0.25, 0.25]))
    result = pfb.score([[0], [1, 1, 1]], pooled, positional)
    rows = (-math.log(0.5) + -math.log(0.25)) / 2
    tokens = (-math.log(0.5) + 3 * -math.log(0.25)) / 4
    assert result["pooled"]["row_mean_ce"] == pytest.approx(rows)
    assert result["pooled"]["token_mean_ce"] == pytest.approx(tokens)


def test_rejects_rows_longer_than_the_canvas():
    counts = np.ones((2, 3), dtype=np.int64)
    pooled, positional = pfb.log_probabilities(counts, 1.0)
    with pytest.raises(ValueError):
        pfb.score([[0, 1, 2]], pooled, positional)


def test_cli_writes_lines_below_uniform(tmp_path):
    corpus(tmp_path / "train", seed=2)
    config = tmp_path / "config.json"
    tiny_config().save(config)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps([[5, 5, 6], {"tokens": [5, 7]}]), encoding="utf-8")
    result = pfb.main(["--fit-corpus", str(tmp_path / "train"), "--config", str(config), "--targets", str(targets),
                       "--samples", "40", "--output", str(tmp_path / "line.json")])
    assert result["rows"] == 2 and result["targets"] == 5
    assert result["pooled"]["row_mean_ce"] < result["uniform"] == math.log(tiny_config().vocab_size)
    assert json.loads((tmp_path / "line.json").read_text(encoding="utf-8"))["targets"] == 5
