"""Canvas positions must enter the value path, including the first all-MASK fill."""

import itertools
import json
import random

import pytest
import torch

from nuri.checkpoint import (add_canvas_positions, load_pretrained, load_training_state,
                             save_pretrained, save_training_state)
from nuri.config import MoEConfig, NuriConfig, PRESETS, tiny_config
from nuri.model import NuriModel
from nuri.generation import generate


def legacy_model(moe=None):
    torch.manual_seed(19)
    return NuriModel(tiny_config(canvas_position_encoding="none", moe=moe)).eval()


def memory(model):
    x, valid = model.encode(torch.tensor([[5, 6, 7], [8, 9, 10]]))
    r = model.reason(x, valid, steps=2)
    return x, valid, r, model.generation_memory(x, valid, r)


def canvas(model, length=4):
    ids = torch.full((2, length), model.config.mask_id)
    return ids, torch.ones_like(ids, dtype=torch.bool)


def test_new_presets_opt_in_but_legacy_configuration_stays_unchanged():
    assert NuriConfig.from_dict({}).canvas_position_encoding == "none"
    for factory in PRESETS.values():
        assert factory().canvas_position_encoding == "learned"
    with pytest.raises(ValueError, match="canvas_position_encoding"):
        tiny_config(canvas_position_encoding="rope")


@pytest.mark.parametrize("moe", [None, MoEConfig(num_experts=4, expert_ffn_size=32,
                                                input_layers=(2,), readout_layers=(2,))])
def test_full_mask_symmetry_negative_control_and_position_gradient(moe):
    model = legacy_model(moe)
    ids, valid = canvas(model)
    _, _, _, cache = memory(model)
    old_hidden = model.readout_hidden(ids, valid, cache)
    old_logits = model.vocab_logits(old_hidden)
    assert (old_logits - old_logits[:, :1]).abs().max() < 1e-5
    add_canvas_positions(model, seed=7)
    hidden = model.readout_hidden(ids, valid, cache)
    logits = model.vocab_logits(hidden)
    assert (hidden - hidden[:, :1]).abs().max() > 0.01
    assert (logits - logits[:, :1]).abs().max() > 0.01
    targets = torch.tensor([[11, 12, 13, 2], [14, 15, 16, 2]])
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    grad = model.readout.position_embedding.weight.grad
    assert grad[:4].abs().sum() > 0
    assert torch.count_nonzero(grad[4:]) == 0


def test_padding_and_memory_cache_equivalence():
    model = add_canvas_positions(legacy_model(), seed=7)
    x, valid, r, cached = memory(model)
    ids, canvas_valid = canvas(model)
    with torch.no_grad():
        expected = model.readout_hidden(ids, canvas_valid, cached)
        recomputed = model.generation_memory(x, valid, r)
        assert torch.equal(expected, model.readout_hidden(ids, canvas_valid, recomputed))
        padded_ids = torch.cat([ids, torch.tensor([[31, 32], [33, 34]])], dim=1)
        padded_valid = torch.cat([canvas_valid, torch.zeros((2, 2), dtype=torch.bool)], dim=1)
        actual = model.readout_hidden(padded_ids, padded_valid, cached)[:, :4]
        assert torch.allclose(expected, actual, atol=1e-6)
        assert torch.equal(x, model.encode(torch.tensor([[5, 6, 7], [8, 9, 10]]))[0])
        assert torch.equal(r, model.reason(x, valid, steps=2))


@pytest.mark.parametrize("length", [0, 17])
def test_canvas_bounds(length):
    model = legacy_model()
    _, _, _, cache = memory(model)
    ids, valid = canvas(model, length)
    with pytest.raises(ValueError, match="canvas length"):
        model.readout_hidden(ids, valid, cache)


def test_migration_only_adds_one_table_and_preserves_encoder_rng_and_dtype(tmp_path):
    model = legacy_model().to(torch.bfloat16)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.tensor([[5, 6, 7]])
    before = model.encode(inputs)[0]
    rng = torch.get_rng_state().clone()
    assert add_canvas_positions(model, seed=71) is model
    assert torch.equal(rng, torch.get_rng_state())
    after = model.state_dict()
    assert set(after) - set(original) == {"readout.position_embedding.weight"}
    for name, value in original.items():
        assert torch.equal(value, after[name]), name
        assert value.dtype == after[name].dtype
    assert torch.equal(before, model.encode(inputs)[0])
    table = model.readout.position_embedding
    assert table.weight.dtype == torch.bfloat16 and table.weight.requires_grad
    save_pretrained(model, tmp_path, dtype=torch.bfloat16)
    restored = load_pretrained(tmp_path, dtype=torch.bfloat16)
    assert restored.config.canvas_position_encoding == "learned"
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match="legacy model"):
        add_canvas_positions(model)


def test_legacy_weights_and_exact_resume_accept_only_the_missing_none_field(tmp_path):
    model = legacy_model()
    weights = tmp_path / "weights"
    save_pretrained(model, weights, dtype=torch.float32)
    config = json.loads((weights / "config.json").read_text())
    del config["canvas_position_encoding"]
    (weights / "config.json").write_text(json.dumps(config))
    restored = load_pretrained(weights)
    assert restored.readout.position_embedding is None
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    optimizer = torch.optim.AdamW(model.parameters())
    state_path = tmp_path / "legacy.pt"
    save_training_state(state_path, model, optimizer, step=5)
    state = torch.load(state_path, weights_only=False)
    del state["config"]["canvas_position_encoding"]
    torch.save(state, state_path)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    assert load_training_state(state_path, restored, restored_optimizer)["step"] == 5
    corrected = add_canvas_positions(restored, seed=3)
    fresh_optimizer = torch.optim.AdamW(corrected.parameters())
    with pytest.raises(ValueError, match="canvas_position_encoding"):
        load_training_state(state_path, corrected, fresh_optimizer)


def test_position_seed_is_local_and_repeatable():
    first = add_canvas_positions(legacy_model(), seed=8)
    second = add_canvas_positions(legacy_model(), seed=8)
    assert torch.equal(first.readout.position_embedding.weight,
                       second.readout.position_embedding.weight)
    with pytest.raises(ValueError, match="integer"):
        add_canvas_positions(legacy_model(), seed=True)


def test_full_mask_learning_is_conditioned_on_input_and_generalizes_combinations():
    """Recover different equal-length sequences, so a position prior cannot pass."""
    torch.manual_seed(7301)
    model = NuriModel(tiny_config(vocab_size=64, max_input_len=8, max_response_len=4))
    combinations = list(itertools.permutations(range(10, 16), 3))
    random.Random(7301).shuffle(combinations)
    inputs = torch.tensor([[6, a, 7, b, 8, c] for a, b, c in combinations])
    targets = torch.tensor([[a + 20, b + 20, c + 20, 2] for a, b, c in combinations])
    assert not torch.isin(inputs, targets.unique()).any()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    batch_rng = torch.Generator().manual_seed(7302)
    for _ in range(400):
        rows = torch.randint(84, (24,), generator=batch_rng)
        x, valid = model.encode(inputs[rows])
        r = model.reason(x, valid, steps=2)
        cache = model.generation_memory(x, valid, r)
        masked = torch.full_like(targets[rows], model.config.mask_id)
        hidden = model.readout_hidden(masked, torch.ones_like(masked, dtype=torch.bool), cache)
        loss = torch.nn.functional.cross_entropy(model.vocab_logits(hidden).flatten(0, 1),
                                                 targets[rows].flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        # One round only: the model was trained on fully masked canvases,
        # which is exactly what round 1 sees. Later refinement rounds see
        # partly filled canvases that this test never trains on, so their
        # accuracy depends on the seed and the platform (seed 7301 on Linux
        # torch 2.11 drops from 1.0 at one round to 0.85 at four); real
        # training mixes mask ratios (corrupt_canvas) for those rounds.
        trained = generate(model, inputs[:84], steps=2, rounds=1, canvas_len=4)
        assert (trained.canvas == targets[:84]).all(1).float().mean() >= 0.95
        result = generate(model, inputs[84:], steps=2, rounds=1, canvas_len=4)
        # Use the reviewed structural prototype bound, not a language
        # quality target. Scratch generalization depends on initialization.
        assert (result.canvas == targets[84:]).all(1).float().mean() >= 0.80
        swapped = generate(model, inputs[84:].roll(1, 0), steps=2, rounds=1, canvas_len=4)
        assert (swapped.canvas == targets[84:].roll(1, 0)).all(1).float().mean() >= 0.80
        assert (swapped.canvas == targets[84:]).all(1).float().mean() <= 0.1
