import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

transformers = pytest.importorskip("transformers")

from nuri.checkpoint import load_pretrained, save_pretrained  # noqa: E402
from nuri.config import MoEConfig, tiny_config  # noqa: E402
from nuri.model import NuriModel  # noqa: E402


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "export_release.py"
    spec = importlib.util.spec_from_file_location("export_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stage", ["base", "it", "embed"])
def test_export_loads_with_transformers(tmp_path, stage):
    torch.manual_seed(0)
    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,), readout_layers=(2,))
    config = tiny_config(moe=moe if stage == "base" else None, length_head_trained=stage == "it")
    save_pretrained(NuriModel(config), tmp_path / "ckpt", dtype=torch.float32)

    out = tmp_path / f"release-{stage}"
    load_script().main(["--checkpoint", str(tmp_path / "ckpt"), "--out", str(out), "--name", f"nuri-{stage}",
                        "--stage", stage, "--tokens", "1e6", "--gpu", "CPU", "--gpu-hours", "0"])

    report = json.loads((tmp_path / f"release-{stage}.verify.json").read_text())
    assert report["ok"]
    hf_config = json.loads((out / "config.json").read_text())
    assert hf_config["model_type"] == "nuri" and "AutoModelForMaskedLM" in hf_config["auto_map"]
    with safe_open(str(out / "model.safetensors"), "pt") as weights:
        assert all(name.startswith("model.") for name in weights.keys())
    card = (out / "README.md").read_text()
    assert "license: mit" in card and f"nuri-{stage}" in card
    assert load_pretrained(out).config.length_head_trained == (stage == "it")  # Nuri loader reads it too


def test_sequence_classification_head_is_new(tmp_path):
    from transformers import AutoModelForSequenceClassification

    save_pretrained(NuriModel(tiny_config()), tmp_path / "ckpt", dtype=torch.float32)
    out = tmp_path / "release"
    load_script().main(["--checkpoint", str(tmp_path / "ckpt"), "--out", str(out), "--name", "n", "--stage", "base"])
    model = AutoModelForSequenceClassification.from_pretrained(str(out), trust_remote_code=True, num_labels=3)
    ids = torch.randint(3, 128, (2, 5))
    output = model(input_ids=ids, labels=torch.tensor([0, 2]))
    assert output.logits.shape == (2, 3) and torch.isfinite(output.loss)


def test_it_export_requires_instruction_tuned_checkpoint(tmp_path):
    save_pretrained(NuriModel(tiny_config()), tmp_path / "ckpt")
    with pytest.raises(SystemExit):
        load_script().main(["--checkpoint", str(tmp_path / "ckpt"), "--out", str(tmp_path / "o"), "--name", "n",
                            "--stage", "it"])


def test_hf_training_forward_adds_and_clears_moe_aux(tmp_path):
    from transformers import AutoModelForMaskedLM

    moe = MoEConfig(num_experts=4, expert_ffn_size=32, top_k=2, input_layers=(2,), readout_layers=(2,))
    save_pretrained(NuriModel(tiny_config(moe=moe)), tmp_path / "ckpt", dtype=torch.float32)
    load_script().main(["--checkpoint", str(tmp_path / "ckpt"), "--out", str(tmp_path / "r"), "--name", "n",
                        "--stage", "base"])
    model = AutoModelForMaskedLM.from_pretrained(str(tmp_path / "r"), trust_remote_code=True,
                                                 dtype=torch.float32).train()
    ids = torch.randint(3, 128, (2, 6))
    plain = torch.nn.functional.cross_entropy(model(input_ids=ids).logits.flatten(0, 1), ids.flatten())
    for _ in range(3):
        output = model(input_ids=ids, labels=ids)
        output.loss.backward()
    assert output.loss > plain.detach()  # balance + z loss were added
    assert all(not layer.aux_records for layer in model.model.moe_layers())
