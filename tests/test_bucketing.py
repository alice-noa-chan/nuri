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
        chosen = length_window(list(range(200)), sizes, input_budget=8 * 32, canvas_budget=8 * 16,
                               max_rows=32, rng=rng)
        rows = [sizes[i] for i in chosen]
        assert chosen and len(rows) <= 32
        assert len(rows) * max(a for a, _ in rows) <= 8 * 32
        assert len(rows) * max(b for _, b in rows) <= 8 * 16


def test_short_inputs_cannot_inflate_the_canvas():
    # Input width 1 with canvas width 159: one combined budget of 64 x (1024 + 256)
    # would accept 512 rows and 81,408 canvas positions instead of 16,384.
    sizes = [(1, 159)] * 600
    chosen = length_window(list(range(600)), sizes, 64 * 1024, 64 * 256, 64 * 4, random.Random(0))
    assert len(chosen) * 159 <= 64 * 256


def test_window_is_a_pure_function_of_the_rng_state():
    sizes = [(i % 30 + 1, i % 15 + 1) for i in range(100)]
    a, b = random.Random(7), random.Random(7)
    for _ in range(10):
        assert length_window(list(range(100)), sizes, 300, 150, 64, a) == length_window(list(range(100)), sizes, 300, 150, 64, b)


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
