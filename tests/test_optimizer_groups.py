from nuri.config import tiny_config
from nuri.model import NuriModel
from nuri.trainer import TrainSettings, make_optimizer


def decayed_names(model, tmp_path):
    optimizer = make_optimizer(
        model, TrainSettings(str(tmp_path), total_steps=10, mix={"mlm": 1.0})
    )
    decayed = {
        id(p)
        for g in optimizer.param_groups
        if g["weight_decay"] > 0
        for p in g["params"]
    }
    return {name for name, p in model.named_parameters() if id(p) in decayed}


def test_position_tables_and_embeddings_are_not_decayed(tmp_path):
    model = NuriModel(
        tiny_config(memory_position_encoding="learned", draft_feedback=True)
    )
    decayed = decayed_names(model, tmp_path)
    for name in (
        "readout.memory_positions.from_start.weight",
        "readout.memory_positions.from_end.weight",
        "readout.position_embedding.weight",
        "draft_positions",
        "embedding.weight",
        "core.slots",
    ):
        assert name in dict(model.named_parameters()), name
        assert name not in decayed, name
    # Ordinary matrices still decay.
    assert "readout.blocks.0.cross_attn.q_proj.weight" in decayed
    assert "encoder.blocks.0.ffn.up_proj.weight" in decayed
