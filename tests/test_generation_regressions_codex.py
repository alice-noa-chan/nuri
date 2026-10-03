"""Independent regressions for generation boundaries and refinement behavior."""

import math

import pytest
import torch

from nuri.config import tiny_config
from nuri.generation import generate
from nuri.model import NuriModel
from nuri.readout import MemoryCache


def small_config():
    return tiny_config(
        vocab_size=8,
        hidden_size=8,
        num_heads=2,
        ffn_size=8,
        input_layers=1,
        readout_layers=2,
        latent_slots=2,
        max_input_len=8,
        max_response_len=8,
        embed_dim=4,
        default_depth=1,
        refine_rounds=3,
    )


class ScriptedModel(NuriModel):
    """Provide deterministic denoising predictions without depending on training.

    Position 1 initially predicts EOS with low confidence, then changes to a
    word. Position 0 is retained in round 2 but must be reconsidered in round
    3 using its stored round-1 confidence. Its hypothetical visible-token
    confidence in round 2 is deliberately higher.
    """

    predictions = (
        ((3, 0.90), (2, 0.30), (5, 0.60), (6, 0.80)),
        ((7, 0.999), (4, 0.95), (5, 0.70), (6, 0.99)),
        ((3, 0.85), (7, 0.99), (2, 0.85), (7, 0.99)),
    )

    def __init__(self, prefer_control_tokens=False):
        super().__init__(small_config())
        self.prefer_control_tokens = prefer_control_tokens
        self.encode_calls = 0
        self.reason_calls = 0
        self.memory_calls = 0
        self.readout_inputs = []
        self.projected_positions = []
        self.cache = object()

    def encode(self, input_ids, attention_mask=None):
        self.encode_calls += 1
        valid = input_ids != self.config.pad_id
        x = torch.zeros(*input_ids.shape, self.config.hidden_size, device=input_ids.device)
        return x, valid

    def reason(self, x, valid, steps=None, return_states=False):
        self.reason_calls += 1
        return x.new_zeros(x.shape[0], self.config.latent_slots, self.config.hidden_size)

    def generation_memory(self, x, valid, r):
        self.memory_calls += 1
        return self.cache

    def readout_hidden(self, canvas_ids, canvas_valid, cache):
        assert cache is self.cache
        self.readout_inputs.append(canvas_ids.clone())
        positions = torch.arange(canvas_ids.shape[1], device=canvas_ids.device)
        return positions[None, :, None].expand(canvas_ids.shape[0], -1, 1).float()

    def vocab_logits(self, hidden):
        positions = hidden[:, 0].long().tolist()
        self.projected_positions.append(positions)
        round_predictions = self.predictions[min(len(self.readout_inputs) - 1, 2)]
        logits = hidden.new_full((len(positions), self.config.vocab_size), -torch.inf)
        allowed_ids = range(2, self.config.vocab_size)
        for row, position in enumerate(positions):
            token, probability = round_predictions[position]
            other_probability = (1.0 - probability) / (self.config.vocab_size - 3)
            for candidate in allowed_ids:
                logits[row, candidate] = math.log(probability if candidate == token else other_probability)
        if self.prefer_control_tokens:
            logits[:, self.config.pad_id] = 100.0
            logits[:, self.config.mask_id] = 99.0
        return logits


@pytest.mark.parametrize("rounds", [0, -1])
def test_nonpositive_rounds_are_rejected(rounds):
    model = ScriptedModel().eval()
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[3, 4]]), canvas_len=4, rounds=rounds)


@pytest.mark.parametrize("max_len", [0, -2, 2.5])
def test_invalid_canvas_cap_is_rejected(max_len):
    model = ScriptedModel().eval()
    with pytest.raises((ValueError, TypeError)):
        generate(model, torch.tensor([[3, 4]]), canvas_len=4, max_len=max_len)


@pytest.mark.parametrize(
    "canvas_len",
    [
        0,
        -3,
        2.5,
        torch.tensor(2.5),
        torch.tensor([2.5, 3.0]),
        torch.tensor([3]),
        torch.tensor([3, 4, 5]),
        torch.tensor([[3], [4]]),
    ],
    ids=["zero", "negative", "fractional", "fractional-scalar-tensor", "fractional-vector",
         "short-vector", "long-vector", "matrix"],
)
def test_invalid_explicit_lengths_are_rejected(canvas_len):
    model = ScriptedModel().eval()
    ids = torch.tensor([[3, 4], [4, 5]])
    with pytest.raises((ValueError, TypeError)):
        generate(model, ids, canvas_len=canvas_len)


@pytest.mark.parametrize("prefer_control_tokens", [False, True])
def test_refinement_revises_eos_and_uses_last_masked_confidence(prefer_control_tokens):
    model = ScriptedModel(prefer_control_tokens).eval()
    result = generate(model, torch.tensor([[3, 4]]), canvas_len=4, rounds=3)

    # An early EOS must not end decoding, and an old visible token can be remasked.
    assert len(model.readout_inputs) == 3
    assert model.readout_inputs[1].tolist() == [[3, 1, 1, 1]]
    assert model.readout_inputs[2].tolist() == [[1, 4, 1, 6]]
    assert result.canvas.tolist() == [[3, 4, 2, 6]]
    assert result.tokens == [[3, 4]]
    assert result.ended_with_eos == [True]

    # Project only current MASK positions; source and memory are prepared once.
    assert model.projected_positions == [[0, 1, 2, 3], [1, 2, 3], [0, 2]]
    assert (model.encode_calls, model.reason_calls, model.memory_calls) == (1, 1, 1)


@pytest.mark.parametrize("cache_layer_count", [0, 1, 3])
def test_readout_rejects_missing_or_extra_cache_layers(cache_layer_count):
    model = NuriModel(small_config()).eval()
    ids = torch.tensor([[3, 4, 5]])
    x, valid = model.encode(ids)
    cache = model.generation_memory(x, valid, model.reason(x, valid))
    malformed = MemoryCache((cache.kv * 2)[:cache_layer_count], cache.mask)
    canvas = torch.full((1, 4), model.config.mask_id)

    with pytest.raises(ValueError):
        model.readout_hidden(canvas, torch.ones_like(canvas, dtype=torch.bool), malformed)
