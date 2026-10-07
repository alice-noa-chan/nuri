"""Standalone releases remain usable without optional CUDA kernel packages."""

import subprocess
import sys

import pytest

pytest.importorskip("transformers")

from transformers.dynamic_module_utils import get_relative_import_files

from nuri.config import tiny_config
from nuri.model import NuriModel
from scripts.export_release import write_folder


def test_export_copies_transitive_adapters_and_loads_with_optional_packages_blocked(
    tmp_path,
):
    model = NuriModel(
        tiny_config(
            hidden_size=16,
            num_heads=2,
            ffn_size=32,
            input_layers=1,
            readout_layers=1,
            embed_dim=8,
        )
    )
    out = tmp_path / "release"
    write_folder(model, out, "base", None)
    dependencies = get_relative_import_files(str(out / "modeling_nuri.py"))
    for name in ("attention_backend.py", "liger_backend.py"):
        assert (out / name).exists()
        assert str(out / name) in dependencies
    # A fresh interpreter cannot reuse this checkout's already imported modules.
    script = """
import importlib.abc
import sys
import torch

blocked = {'flash_attn', 'flash_attn_3', 'flash_attn_interface', 'liger_kernel'}
original_find_spec = importlib.util.find_spec
def without_optional(name, package=None):
    if name.split('.')[0] in blocked:
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = without_optional

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('optional GPU package is blocked by this test')

sys.meta_path.insert(0, BlockOptional())
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(sys.argv[1], trust_remote_code=True).float().eval()
model.model.set_attention_backend('sdpa')
model.model.set_liger_kernels(False)
with torch.no_grad():
    logits = model(input_ids=torch.tensor([[5, 6, 7]])).logits
assert logits.shape == (1, 3, 128) and torch.isfinite(logits).all()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(out)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
