import pytest
import torch

from nuri.config import tiny_config
from nuri.encoder import local_mask
from nuri.model import NuriModel


def test_local_mask_band_padding_and_self():
    valid = torch.tensor([[True, True, True, False, False]])
    mask = local_mask(valid, radius=1)[0]
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[2].tolist() == [False, True, True, False, False]  # padding key 3 is hidden
    assert mask[4].tolist() == [False, False, False, False, True]  # a padding query sees only itself


def test_local_layers_ignore_far_tokens():
    torch.manual_seed(0)
    model = NuriModel(tiny_config(input_layers=1, attention_schedule="L", local_radius=2)).eval()
    ids = torch.randint(3, 128, (1, 12))
    x, _ = model.encode(ids)
    far = ids.clone()
    far[0, 8:] = torch.randint(3, 128, (4,))  # more than 2 positions away from position 0..5
    x_far, _ = model.encode(far)
    assert torch.allclose(x[0, :6], x_far[0, :6], atol=1e-6)
    assert not torch.allclose(x[0, 6:], x_far[0, 6:], atol=1e-4)


def test_mixed_schedule_stays_finite_with_long_padding():
    torch.manual_seed(0)
    model = NuriModel(tiny_config(input_layers=2, attention_schedule="GL", local_radius=1)).eval()
    ids = torch.full((2, 16), model.config.pad_id)
    ids[0, :3] = torch.tensor([5, 6, 7])
    ids[1, :16] = torch.randint(3, 128, (16,))
    x, valid = model.encode(ids)
    assert torch.isfinite(x).all()
    short, _ = model.encode(ids[:1, :3])
    assert torch.allclose(x[0, :3], short[0], atol=1e-5)  # padding changes nothing


def test_global_schedule_matches_default():
    torch.manual_seed(0)
    default = NuriModel(tiny_config()).eval()
    torch.manual_seed(0)
    explicit = NuriModel(tiny_config(attention_schedule="GG")).eval()
    ids = torch.randint(3, 128, (2, 7))
    assert torch.equal(default.encode(ids)[0], explicit.encode(ids)[0])


@pytest.mark.parametrize("schedule", ["G", "GLX", "GX"])
def test_bad_schedule_is_rejected(schedule):
    with pytest.raises(ValueError):
        tiny_config(attention_schedule=schedule)
