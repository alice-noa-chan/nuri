"""Configuration errors should fail before tensors or optimizers are allocated."""

import pytest

from nuri.config import MoEConfig, NuriConfig


@pytest.mark.parametrize(
    "field",
    [
        "vocab_size",
        "hidden_size",
        "num_heads",
        "ffn_size",
        "input_layers",
        "readout_layers",
        "latent_slots",
        "max_input_len",
        "max_response_len",
        "embed_dim",
        "default_depth",
        "refine_rounds",
        "num_labels",
        "pad_id",
        "mask_id",
        "eos_id",
        "local_radius",
        "slot_groups",
        "num_kv_heads",
        "kv_latent_dim",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, float("nan")])
def test_noninteger_geometry_is_rejected(field, value):
    with pytest.raises(ValueError, match="integer"):
        NuriConfig(**{field: value})


@pytest.mark.parametrize(
    "field", ["rope_theta", "norm_eps", "gate_init", "initializer_range"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_model_constants_are_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        NuriConfig(**{field: value})


@pytest.mark.parametrize("depths", [(True,), (1.5,), (float("nan"),)])
def test_noninteger_training_depth_is_rejected(depths):
    with pytest.raises(ValueError, match="integer"):
        NuriConfig(train_depths=depths)


@pytest.mark.parametrize(
    "field", ["num_experts", "expert_ffn_size", "top_k", "num_shared_experts"]
)
def test_noninteger_moe_geometry_is_rejected(field):
    with pytest.raises(ValueError, match="integer"):
        NuriConfig(
            moe=MoEConfig(input_layers=(1,), readout_layers=(1,), **{field: 2.5})
        )


@pytest.mark.parametrize("field", ["balance_loss_weight", "z_loss_weight"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_moe_loss_weight_is_rejected(field, value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        NuriConfig(
            moe=MoEConfig(input_layers=(1,), readout_layers=(1,), **{field: value})
        )


def test_fractional_moe_layer_index_is_rejected():
    with pytest.raises(ValueError, match="indices"):
        NuriConfig(moe=MoEConfig(input_layers=(1.5,), readout_layers=(1,)))
