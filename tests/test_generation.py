import math

import pytest
import torch

from nuri.config import tiny_config
from nuri.generation import generate
from nuri.model import NuriModel


def make_model(**overrides):
    torch.manual_seed(0)
    return NuriModel(tiny_config(**overrides)).eval()


def prompt(model):
    return torch.randint(3, model.config.vocab_size, (2, 7))


def test_base_model_needs_explicit_length():
    model = make_model()
    with pytest.raises(ValueError):
        generate(model, prompt(model))


def test_remask_schedule_and_final_canvas():
    model = make_model()
    rounds, length = 4, 10
    result = generate(model, prompt(model), rounds=rounds, canvas_len=length, keep_history=True)
    mask_id = model.config.mask_id
    for s, canvas in enumerate(result.history, start=1):
        expected = math.ceil(length * (rounds - s) / rounds)
        assert (canvas == mask_id).sum(dim=1).tolist() == [expected, expected]
    assert not (result.canvas == mask_id).any()


def test_per_sample_lengths_and_eos_trimming():
    model = make_model()
    eos = model.config.eos_id
    # Force the vocabulary head to prefer EOS: every answer must be empty.
    original = model.vocab_logits

    def eos_logits(hidden):
        logits = original(hidden)
        logits[:, eos] = 1e4
        return logits

    model.vocab_logits = eos_logits
    result = generate(model, prompt(model), canvas_len=torch.tensor([3, 5]))
    assert result.lengths.tolist() == [3, 5]
    assert result.tokens == [[], []] and result.ended_with_eos == [True, True]
    assert (result.canvas[0, 3:] == model.config.pad_id).all()


def test_length_head_used_after_instruction_tuning():
    model = make_model(length_head_trained=True)
    result = generate(model, prompt(model), max_len=6)
    assert (result.lengths >= 1).all() and (result.lengths <= 6).all()


def test_deterministic():
    model = make_model()
    ids = prompt(model)
    a = generate(model, ids, canvas_len=8)
    b = generate(model, ids, canvas_len=8)
    assert torch.equal(a.canvas, b.canvas)
