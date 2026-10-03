import torch

from nuri.config import MoEConfig, reference_config, release_dense_config, release_moe_config, tiny_config
from nuri.model import NuriModel

# Small parameters that ARCHITECTURE.md §10 leaves out of the headline counts.
SMALL_PARAMS = ("norm", "gate_logits", "length_head", "task_head")


def main_weight_count(config) -> int:
    with torch.device("meta"):
        model = NuriModel(config)
    return sum(p.numel() for name, p in model.named_parameters() if not any(s in name for s in SMALL_PARAMS))


def test_parameter_counts_match_architecture_doc():
    assert main_weight_count(reference_config()) == 49_225_728
    assert main_weight_count(release_dense_config()) == 150_233_088
    assert main_weight_count(release_moe_config()) == 242_319_360


def make_model(**overrides):
    torch.manual_seed(0)
    return NuriModel(tiny_config(**overrides)).eval()


def sample_inputs(config, batch=2, length=10):
    ids = torch.randint(3, config.vocab_size, (batch, length))
    ids[0, -3:] = config.pad_id  # first sample is shorter
    return ids


def test_canvas_never_changes_memory_or_reasoning():
    """Information contract (§7): X and R must not see the output canvas."""
    model = make_model()
    ids = sample_inputs(model.config)
    x, valid = model.encode(ids)
    r = model.reason(x, valid, steps=2)
    cache = model.generation_memory(x, valid, r)
    canvas = torch.full((2, 6), model.config.mask_id)
    canvas_valid = torch.ones(2, 6, dtype=torch.bool)
    h1 = model.readout_hidden(canvas, canvas_valid, cache)
    other = torch.randint(3, model.config.vocab_size, (2, 6))
    h2 = model.readout_hidden(other, canvas_valid, cache)
    assert not torch.allclose(h1, h2)
    x2, _ = model.encode(ids)
    assert torch.equal(x, x2) and torch.equal(r, model.reason(x2, valid, steps=2))


def test_cached_memory_matches_fresh_memory():
    model = make_model()
    ids = sample_inputs(model.config)
    x, valid = model.encode(ids)
    r = model.reason(x, valid, steps=2)
    canvas = torch.randint(3, model.config.vocab_size, (2, 5))
    canvas_valid = torch.ones(2, 5, dtype=torch.bool)
    cache = model.generation_memory(x, valid, r)
    first = model.readout_hidden(canvas, canvas_valid, cache)
    second = model.readout_hidden(canvas, canvas_valid, cache)  # reused cache
    fresh = model.readout_hidden(canvas, canvas_valid, model.generation_memory(x, valid, r))
    assert torch.equal(first, second) and torch.allclose(first, fresh)


def test_canvas_padding_is_invisible():
    model = make_model()
    ids = sample_inputs(model.config)
    x, valid = model.encode(ids)
    cache = model.generation_memory(x, valid, model.reason(x, valid, steps=1))
    canvas = torch.randint(3, model.config.vocab_size, (2, 6))
    canvas_valid = torch.ones(2, 6, dtype=torch.bool)
    canvas_valid[0, 4:] = False
    h = model.readout_hidden(canvas, canvas_valid, cache)
    canvas2 = canvas.clone()
    canvas2[0, 4:] = 5
    assert torch.allclose(h[0, :4], model.readout_hidden(canvas2, canvas_valid, cache)[0, :4], atol=1e-6)


def test_readout_loss_trains_every_part():
    model = make_model().train()
    ids = sample_inputs(model.config)
    x, valid = model.encode(ids)
    r = model.reason(x, valid, steps=2)
    hidden = model.readout_hidden(torch.full((2, 4), model.config.mask_id), torch.ones(2, 4, dtype=torch.bool),
                                  model.generation_memory(x, valid, r))
    model.vocab_logits(hidden).logsumexp(-1).mean().backward()
    for module in (model.embedding, model.encoder.blocks[0].attn.q_proj, model.core.cross_attn.q_proj,
                   model.readout.blocks[-1].ffn.down_proj):
        assert module.weight.grad.abs().sum() > 0


def test_heads_and_embeddings():
    model = make_model(num_labels=3)
    config = model.config
    ids = sample_inputs(config)
    x, valid = model.encode(ids)
    u, r = model.pool(x, valid, model.reason(x, valid))
    assert model.length_logits(u, r).shape == (2, config.max_response_len)
    assert model.task_logits(u, r).shape == (2, 3)
    select = torch.zeros_like(valid)
    select[:, 1] = True
    assert model.mlm_logits(x, select).shape == (2, config.vocab_size)
    for mode in ("fast", "reasoning"):
        z = model.embed(ids, mode=mode, dim=16)
        assert z.shape == (2, 16) and torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_moe_aux_is_collected_once():
    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,), readout_layers=(2,))
    model = make_model(moe=moe)
    ids = sample_inputs(model.config)
    x, valid = model.encode(ids)
    aux = model.pop_moe_aux()
    assert aux["loss"] is not None and len(aux["counts"]) == 1
    assert model.pop_moe_aux()["loss"] is None
