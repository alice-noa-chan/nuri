import pytest

from nuri.config import PRESETS, MoEConfig, NuriConfig


def test_presets_build():
    for name, factory in PRESETS.items():
        config = factory()
        assert config.hidden_size % config.num_heads == 0, name


def test_release_overrides_match_architecture():
    dense = PRESETS["release-dense"]()
    assert (dense.hidden_size, dense.num_heads, dense.ffn_size) == (1024, 16, 2816)
    assert (dense.input_layers, dense.latent_slots, dense.max_response_len) == (14, 32, 256)
    assert dense.qk_norm and dense.train_depths == (2, 4, 8)
    moe = PRESETS["release-moe"]()
    active_width = moe.moe.expert_ffn_size * (moe.moe.top_k + moe.moe.num_shared_experts)
    assert active_width == dense.ffn_size
    assert moe.moe.input_layers == (2, 4, 6, 8, 10, 12, 14)  # every second layer


def test_roundtrip(tmp_path):
    config = PRESETS["release-moe"]()
    path = tmp_path / "config.json"
    config.save(path)
    loaded = NuriConfig.load(path)
    assert loaded == config
    assert isinstance(loaded.moe, MoEConfig)


@pytest.mark.parametrize("overrides", [
    dict(hidden_size=100, num_heads=8),
    dict(ffn_size=0),
    dict(train_depths=()),
    dict(train_depths=(2, -1)),
    dict(gate_init=1.0),
    dict(norm_eps=0.0),
    dict(mask_id=0),
    dict(eos_id=40000),
    dict(moe=MoEConfig(top_k=1)),
    dict(moe=MoEConfig(input_layers=(7,))),
    dict(moe=MoEConfig(readout_layers=(3,))),
])
def test_rejects_invalid_configs(overrides):
    with pytest.raises(ValueError):
        NuriConfig(**overrides)
