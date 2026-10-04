import random

import torch

from nuri.config import tiny_config
from nuri.data import TokenCorpus, length_window, write_corpus
from nuri.model import NuriModel
from nuri.pipeline import base_stage, instruction_stage
from nuri.trainer import TrainSettings, train

CONFIG = tiny_config()  # N = 32, M = 16


def mixed_documents(rng, count):
    """Mostly short documents with a few long ones, like real text."""
    return [[rng.randrange(3, 120) for _ in range(rng.choice([4, 6, 8, 40]))] for _ in range(count)]


def test_window_fits_the_budget_and_keeps_similar_lengths():
    rng = random.Random(0)
    sizes = [(rng.randrange(1, 33), rng.randrange(1, 17)) for _ in range(200)]
    for _ in range(50):
        chosen = length_window(list(range(200)), sizes, budget=8 * 48, max_rows=64, rng=rng)
        rows = [sizes[i] for i in chosen]
        padded = len(rows) * (max(a for a, _ in rows) + max(b for _, b in rows))
        assert chosen and (padded <= 8 * 48 or len(rows) == 1)
        assert len(rows) <= 64


def test_window_is_a_pure_function_of_the_rng_state():
    sizes = [(i % 30 + 1, i % 15 + 1) for i in range(100)]
    a, b = random.Random(7), random.Random(7)
    for _ in range(10):
        assert length_window(list(range(100)), sizes, 400, 64, a) == length_window(list(range(100)), sizes, 400, 64, b)


def test_bucketed_batches_have_less_padding(tmp_path):
    write_corpus(tmp_path / "corpus", mixed_documents(random.Random(0), 400))
    corpus = TokenCorpus(tmp_path / "corpus")

    def valid_share(bucket):
        span = base_stage(corpus, CONFIG, batch_size=8, seed=1, bucket=bucket)[0]["span"]
        valid = total = 0
        for _ in range(30):
            b = span.next_batch()
            valid += int(b["input_valid"].sum() + b["canvas_valid"].sum())
            total += b["input_valid"].numel() + b["canvas_valid"].numel()
        return valid / total

    assert valid_share(True) > valid_share(False) + 0.05


def test_bucketed_stages_train_and_resume_the_same_batches(tmp_path):
    rng = random.Random(0)
    write_corpus(tmp_path / "corpus", mixed_documents(rng, 200))
    corpus = TokenCorpus(tmp_path / "corpus")
    examples = [([rng.randrange(3, 120) for _ in range(rng.randrange(2, 30))],
                 [rng.randrange(3, 120) for _ in range(rng.randrange(1, 15))]) for _ in range(100)]
    tasks, mix = instruction_stage(examples, CONFIG, batch_size=4, corpus=corpus, seed=0, bucket=True)
    assert tasks["instruction"].options["bucket"] and tasks["span"].options["bucket"]

    # Restoring the random sources reproduces the next batch exactly (the resume contract).
    for task in tasks.values():
        task.next_batch()
        saved = (task.rng.getstate(), task.generator.get_state())
        first = task.next_batch()
        task.rng.setstate(saved[0]), task.generator.set_state(saved[1])
        again = task.next_batch()
        assert all(torch.equal(first[k], again[k]) for k in first if torch.is_tensor(first[k]))

    torch.manual_seed(0)
    settings = TrainSettings(out_dir=str(tmp_path / "run"), total_steps=3, peak_lr=1e-3, warmup_steps=1,
                             mix=mix, progress=False)
    assert train(NuriModel(CONFIG), tasks, settings, device=torch.device("cpu"))
