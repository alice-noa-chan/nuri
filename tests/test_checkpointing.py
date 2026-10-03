import torch

from nuri.config import MoEConfig, tiny_config
from nuri.data import build_instruction_batch
from nuri.model import NuriModel
from nuri.trainer import canvas_loss

MOE = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,), readout_layers=(2,))


def gradients(checkpointing: bool, moe=None):
    torch.manual_seed(0)
    model = NuriModel(tiny_config(moe=moe)).train()
    model.set_gradient_checkpointing(checkpointing)
    batch = build_instruction_batch([([5, 6, 7], [8, 9]), ([10, 11], [12])], model.config,
                                    torch.Generator().manual_seed(0))
    loss = canvas_loss(model, batch, depth=2)["loss"]
    aux = model.pop_moe_aux()
    if aux["loss"] is not None:
        loss = loss + aux["loss"]
    loss.backward()
    leftover = sum(len(m.aux_records) for m in model.moe_layers())
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, aux, leftover


def test_checkpointing_keeps_gradients_identical():
    plain, _, _ = gradients(False)
    saved, _, _ = gradients(True)
    assert plain.keys() == saved.keys()
    for name in plain:
        assert torch.allclose(plain[name], saved[name], atol=1e-6), name


def test_moe_aux_is_not_recorded_twice_by_recomputation():
    plain, aux_plain, _ = gradients(False, MOE)
    saved, aux_saved, leftover = gradients(True, MOE)
    assert leftover == 0  # backward recomputation added no records
    assert torch.allclose(aux_plain["loss"], aux_saved["loss"])
    for name in plain:
        assert torch.allclose(plain[name], saved[name], atol=1e-6), name


def test_no_checkpointing_at_inference():
    model = NuriModel(tiny_config()).eval()
    model.set_gradient_checkpointing(True)
    with torch.no_grad():
        x, _ = model.encode(torch.randint(3, 128, (1, 5)))
    assert x.shape == (1, 5, model.config.hidden_size)
