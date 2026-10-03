import pytest

from nuri.config import PRESETS, MoEConfig, NuriConfig


def test_presets_build():
    for name, factory in PRESETS.items():
        config = factory()
        assert config.hidden_size % config.num_heads == 0, name


def test_release_overrides_match_architecture():
    dense = PRESETS["release-dense"]()
    assert (dense.hidden_size, dense.num_heads, dense.ffn_size) == (768, 12, 2048)
    assert (dense.input_layers, dense.latent_slots, dense.max_response_len) == (10, 32, 256)
    assert dense.qk_norm and dense.train_depths == (2, 4, 8)
    moe = PRESETS["release-moe"]()
    active_width = moe.moe.expert_ffn_size * (moe.moe.top_k + moe.moe.num_shared_experts)
    assert active_width == dense.ffn_size


def test_roundtrip(tmp_path):
    config = PRESETS["release-moe"]()
    path = tmp_path / "config.json"
    config.save(path)
    loaded = NuriConfig.load(path)
    assert loaded == config
    assert isinstance(loaded.moe, MoEConfig)


def test_rejects_bad_heads():
    with pytest.raises(ValueError):
        NuriConfig(hidden_size=100, num_heads=8)
