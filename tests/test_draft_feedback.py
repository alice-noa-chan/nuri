import pytest
import torch

from nuri.config import tiny_config
from nuri.data import build_instruction_batch
from nuri.generation import generate
from nuri.model import NuriModel
from nuri.trainer import canvas_loss


def make_model(**overrides):
    torch.manual_seed(0)
    return NuriModel(tiny_config(draft_feedback=True, **overrides)).eval()


def test_r_reads_visible_draft_tokens_but_not_hidden_targets():
    model = make_model()
    ids = torch.randint(3, 128, (1, 6))
    x, valid = model.encode(ids)
    canvas = torch.tensor([[40, model.config.mask_id, 42]])
    canvas_valid = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        r = model.reason(x, valid, 2, draft=(canvas, canvas_valid))
        other_visible = canvas.clone()
        other_visible[0, 0] = 77
        assert not torch.allclose(r, model.reason(x, valid, 2, draft=(other_visible, canvas_valid)))
        # The model only ever receives MASK for hidden positions, so a hidden target cannot matter;
        # the plain R without any draft differs from the drafted one.
        assert not torch.allclose(r, model.reason(x, valid, 2))


def test_training_uses_the_draft_and_keeps_the_length_head_blind_to_it():
    model = make_model(length_head_trained=True).train()
    batch = build_instruction_batch([([5, 6, 7], [8, 9]), ([10, 11], [12])], model.config,
                                    torch.Generator().manual_seed(0))
    seen = []
    original = model.length_logits
    model.length_logits = lambda u, r: seen.append(r.detach().clone()) or original(u, r)
    torch.manual_seed(1)
    result = canvas_loss(model, batch, depth=2, length_weight=0.1)
    result["loss"].backward()
    assert model.draft_embedding.grad is not None and model.draft_embedding.grad.abs().sum() > 0
    # The length head got the R computed without the draft (whose shape is the gold length).
    x, valid = model.encode(batch["input_ids"], batch["input_valid"])
    _, blind = model.pool(x, valid, model.reason(x, valid, 2))
    assert torch.allclose(seen[0], blind.detach(), atol=1e-6)


def test_generation_recomputes_reasoning_every_round():
    model = make_model()
    calls = []
    original = model.reason

    def counting_reason(*args, **kwargs):
        calls.append(kwargs.get("draft") is not None)
        return original(*args, **kwargs)

    model.reason = counting_reason
    result = generate(model, torch.randint(3, 128, (2, 5)), canvas_len=4, rounds=3)
    assert calls == [False, True, True, True]  # once for the length/initial R, then once per round
    assert not (result.canvas == model.config.mask_id).any()


def test_draft_requires_the_flag():
    torch.manual_seed(0)
    model = NuriModel(tiny_config()).eval()
    x, valid = model.encode(torch.randint(3, 128, (1, 4)))
    with pytest.raises(ValueError):
        model.reason(x, valid, 1, draft=(torch.tensor([[5]]), torch.ones(1, 1, dtype=torch.bool)))


def test_draft_order_matters_to_reasoning():
    model = make_model()
    with torch.no_grad():
        torch.nn.init.normal_(model.draft_positions, std=0.5)  # trained positions are not all zero
        x, valid = model.encode(torch.randint(3, 128, (1, 6)))
        canvas_valid = torch.ones(1, 3, dtype=torch.bool)
        forward = model.reason(x, valid, 2, draft=(torch.tensor([[40, 41, 42]]), canvas_valid))
        backward = model.reason(x, valid, 2, draft=(torch.tensor([[42, 41, 40]]), canvas_valid))
    assert not torch.allclose(forward, backward, atol=1e-5)
