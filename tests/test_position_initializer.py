"""The explicit repair tool preserves existing checkpoint bytes and tensor values."""

import json

import pytest
import torch

from nuri.checkpoint import load_pretrained, save_pretrained
from nuri.config import tiny_config
from nuri.model import NuriModel
from tools.initialize_canvas_positions import file_hash, initialize, main


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_initializer_is_verified_and_cannot_overwrite(tmp_path, dtype):
    model = NuriModel(tiny_config(canvas_position_encoding="none")).to(dtype)
    source, output = tmp_path / "old", tmp_path / "new"
    receipt = tmp_path / ".private/initialization.json"
    save_pretrained(model, source, dtype=dtype)
    source_hash = file_hash(source / "model.safetensors")
    main(["--source", str(source), "--out", str(output), "--receipt", str(receipt), "--seed", "7"])
    evidence = json.loads(receipt.read_text())
    assert evidence["source_hashes"]["model.safetensors"] == source_hash
    assert evidence["output_hashes"]["model.safetensors"] == file_hash(output / "model.safetensors")
    assert file_hash(source / "model.safetensors") == source_hash
    assert not evidence["exact_resume"] and not evidence["optimizer_state_copied"]
    restored = load_pretrained(output, dtype=dtype)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])
    assert restored.config.canvas_position_encoding == "learned"
    with pytest.raises(ValueError, match="fresh paths"):
        initialize(source, output, receipt)


def test_receipt_requires_private_location(tmp_path):
    with pytest.raises(ValueError, match=".private"):
        initialize(tmp_path / "missing", tmp_path / "new", tmp_path / "receipt.json")
