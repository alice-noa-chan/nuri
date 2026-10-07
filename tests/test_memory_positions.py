import pytest
import torch

from nuri.checkpoint import (
    add_memory_positions,
    load_pretrained,
    save_pretrained,
    save_training_state,
    load_training_state,
)
from nuri.config import NuriConfig, release_dense_config, tiny_config
from nuri.generation import generate
from nuri.model import NuriModel
from nuri.readout import MemoryPositions
from nuri.trainer import canvas_loss


def make_model(**overrides):
    torch.manual_seed(0)
    return NuriModel(tiny_config(**overrides))


def prompts():
    ids = torch.randint(3, 128, (3, 9), generator=torch.Generator().manual_seed(1))
    ids[0, 6:] = 0  # right padding
    return ids


def test_option_is_off_by_default_and_validated():
    assert NuriConfig().memory_position_encoding == "none"
    assert release_dense_config().memory_position_encoding == "none"
    assert make_model().readout.memory_positions is None
    legacy = NuriConfig().to_dict()
    del legacy["memory_position_encoding"]
    assert NuriConfig.from_dict(legacy).memory_position_encoding == "none"
    with pytest.raises(ValueError):
        tiny_config(memory_position_encoding="sinusoidal")


def test_new_tables_start_at_zero():
    tables = make_model(memory_position_encoding="learned").readout.memory_positions
    assert isinstance(tables, MemoryPositions)
    assert not tables.from_start.weight.any() and not tables.from_end.weight.any()


def test_position_indices_follow_right_padding():
    tables = MemoryPositions(max_input_len=8, hidden_size=4)
    with torch.no_grad():
        tables.from_start.weight.copy_(torch.arange(8.0)[:, None].expand(8, 4))
        tables.from_end.weight.copy_(100 * torch.arange(8.0)[:, None].expand(8, 4))
    valid = torch.tensor([[True, True, True, False, False], [True] * 5])
    out = tables(torch.zeros(2, 5, 4), valid)[..., 0]
    # value = index from start + 100 * index from end; padding stays zero
    assert out.tolist() == [
        [200.0, 101.0, 2.0, 0.0, 0.0],
        [400.0, 301.0, 202.0, 103.0, 4.0],
    ]
    with pytest.raises(ValueError):
        tables(torch.zeros(1, 9, 4), torch.ones(1, 9, dtype=torch.bool))


def test_migration_keeps_outputs_and_rejects_a_second_call():
    model = make_model().eval()
    before = generate(model, prompts(), canvas_len=7).canvas
    state = {k: v.clone() for k, v in model.state_dict().items()}
    add_memory_positions(model)
    assert model.config.memory_position_encoding == "learned"
    assert torch.equal(generate(model, prompts(), canvas_len=7).canvas, before)
    for name, tensor in state.items():
        assert torch.equal(model.state_dict()[name], tensor)
    with pytest.raises(ValueError):
        add_memory_positions(model)


def test_canvas_loss_trains_the_tables():
    model = make_model(memory_position_encoding="learned")
    ids = prompts()
    canvas_valid = torch.ones(3, 5, dtype=torch.bool)
    targets = torch.randint(3, 128, (3, 5))
    batch = {
        "input_ids": ids,
        "input_valid": ids != 0,
        "canvas": torch.full((3, 5), model.config.mask_id),
        "canvas_valid": canvas_valid,
        "targets": targets,
        "loss_mask": canvas_valid,
    }
    canvas_loss(model, batch, depth=1)["loss"].backward()
    tables = model.readout.memory_positions
    assert tables.from_start.weight.grad.abs().sum() > 0
    assert tables.from_end.weight.grad.abs().sum() > 0
    # Rows past the longest input (9 tokens) never receive a gradient.
    assert not tables.from_start.weight.grad[9:].any()


def test_tables_change_memory_only_at_real_tokens():
    model = make_model(memory_position_encoding="learned")
    with torch.no_grad():
        model.readout.memory_positions.from_start.weight.normal_()
    ids = prompts()
    valid = ids != 0
    x, _ = model.encode(ids)
    shifted = model.readout.memory_positions(x, valid)
    assert torch.equal(shifted[~valid], x[~valid])
    assert not torch.allclose(shifted[valid], x[valid])


def test_save_load_and_resume_round_trip(tmp_path):
    model = make_model(memory_position_encoding="learned")
    with torch.no_grad():
        model.readout.memory_positions.from_end.weight.normal_()
    save_pretrained(model, tmp_path / "model", dtype=torch.float32)
    loaded = load_pretrained(tmp_path / "model")
    assert loaded.config.memory_position_encoding == "learned"
    assert torch.equal(
        loaded.readout.memory_positions.from_end.weight,
        model.readout.memory_positions.from_end.weight,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    save_training_state(tmp_path / "state.pt", model, optimizer, step=3)
    fresh = make_model(memory_position_encoding="learned")
    load_training_state(
        tmp_path / "state.pt", fresh, torch.optim.AdamW(fresh.parameters())
    )
    other = make_model()
    with pytest.raises(ValueError):
        load_training_state(
            tmp_path / "state.pt", other, torch.optim.AdamW(other.parameters())
        )
