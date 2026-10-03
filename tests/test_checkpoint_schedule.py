import torch

from nuri.checkpoint import load_pretrained, load_training_state, save_pretrained, save_training_state
from nuri.config import MoEConfig, tiny_config
from nuri.model import NuriModel
from nuri.schedule import WSDSchedule


def test_public_weights_roundtrip_with_shared_embedding(tmp_path):
    torch.manual_seed(0)
    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,), readout_layers=(2,))
    model = NuriModel(tiny_config(moe=moe)).eval()
    save_pretrained(model, tmp_path, dtype=torch.float32)
    loaded = load_pretrained(tmp_path).eval()
    assert loaded.embedding is loaded.encoder.embedding  # still one shared tensor
    ids = torch.randint(3, 128, (2, 6))
    x1, valid = model.encode(ids)
    x2, _ = loaded.encode(ids)
    assert torch.equal(x1, x2)
    assert loaded.config == model.config


def test_bf16_export_is_close(tmp_path):
    torch.manual_seed(0)
    model = NuriModel(tiny_config()).eval()
    save_pretrained(model, tmp_path)  # bf16 by default
    loaded = load_pretrained(tmp_path).eval()
    ids = torch.randint(3, 128, (2, 6))
    assert torch.allclose(model.encode(ids)[0], loaded.encode(ids)[0], atol=0.1)


def test_training_state_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = NuriModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.embed(torch.randint(3, 128, (2, 6))).sum().backward()
    optimizer.step()
    save_training_state(tmp_path / "state.pt", model, optimizer, step=7, extra={"tokens": 123})
    fresh = NuriModel(tiny_config())
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    state = load_training_state(tmp_path / "state.pt", fresh, fresh_opt)
    assert state["step"] == 7 and state["extra"]["tokens"] == 123
    assert torch.equal(fresh.embedding.weight, model.embedding.weight)


def test_wsd_schedule_shape():
    schedule = WSDSchedule(peak_lr=1.0, total_steps=100, warmup_steps=10, decay_fraction=0.2)
    lrs = [schedule.lr(s) for s in range(100)]
    assert lrs[0] == 0.1 and lrs[9] == 1.0
    assert all(lr == 1.0 for lr in lrs[10:80])
    assert lrs[99] == 0.0 and all(a > b for a, b in zip(lrs[80:], lrs[81:]))
    assert schedule.is_pre_decay(79) and not schedule.is_pre_decay(80)
