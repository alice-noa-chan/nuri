import torch
from torch import nn

from nuri.config import tiny_config
from nuri.core import RecurrentReasoningCore
from nuri.encoder import InputEncoder


class Parts:
    """Embedding + encoder + core: the first half of NuriModel."""

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.config = tiny_config()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.encoder = InputEncoder(self.config)
        self.core = RecurrentReasoningCore(self.config)

    def encode(self, ids, valid):
        return self.encoder(self.embedding(ids), valid)


def test_padding_does_not_change_valid_outputs():
    parts = Parts()
    config = parts.config
    ids = torch.randint(3, config.vocab_size, (1, 7))
    valid = torch.ones(1, 7, dtype=torch.bool)
    x = parts.encode(ids, valid)
    r = parts.core(x, valid, steps=3)

    padded = torch.cat([ids, torch.randint(3, config.vocab_size, (1, 5))], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(1, 5, dtype=torch.bool)], dim=1)
    xp = parts.encode(padded, padded_valid)
    rp = parts.core(xp, padded_valid, steps=3)
    assert torch.allclose(x, xp[:, :7], atol=1e-5)
    assert torch.allclose(r, rp, atol=1e-5)


def test_samples_do_not_interact():
    parts = Parts()
    config = parts.config
    ids = torch.randint(3, config.vocab_size, (2, 6))
    valid = torch.ones(2, 6, dtype=torch.bool)
    r = parts.core(parts.encode(ids, valid), valid, steps=2)
    ids2 = ids.clone()
    ids2[1] = torch.randint(3, config.vocab_size, (6,))
    r2 = parts.core(parts.encode(ids2, valid), valid, steps=2)
    assert torch.allclose(r[0], r2[0], atol=1e-6)
    assert not torch.allclose(r[1], r2[1], atol=1e-3)


def test_gradient_reaches_encoder_through_every_step():
    parts = Parts()
    ids = torch.randint(3, parts.config.vocab_size, (2, 6))
    valid = torch.ones(2, 6, dtype=torch.bool)
    parts.core(parts.encode(ids, valid), valid, steps=3).sum().backward()
    assert parts.embedding.weight.grad.abs().sum() > 0
    assert parts.encoder.blocks[0].attn.q_proj.weight.grad.abs().sum() > 0
    assert parts.core.slots.grad.abs().sum() > 0 and parts.core.gate_logits.grad.abs().sum() > 0


def test_memory_projected_once_per_forward():
    parts = Parts()
    core = parts.core
    calls = []
    original = core.cross_attn.project_kv
    core.cross_attn.project_kv = lambda *a, **k: calls.append(1) or original(*a, **k)
    x = torch.randn(2, 6, parts.config.hidden_size)
    core(x, torch.ones(2, 6, dtype=torch.bool), steps=5)
    assert len(calls) == 1


def test_depth_changes_state_and_states_are_returned():
    parts = Parts()
    config, core = parts.config, parts.core
    x = torch.randn(2, 6, config.hidden_size)
    valid = torch.ones(2, 6, dtype=torch.bool)
    r1 = core(x, valid, steps=1)
    r4, states = core(x, valid, steps=4, return_states=True)
    assert len(states) == 5 and states[0].shape == (2, config.latent_slots, config.hidden_size)
    assert not torch.allclose(r1, r4)
    gate = torch.sigmoid(core.gate_logits)
    assert torch.allclose(gate, torch.full((3,), 0.1), atol=1e-6)
