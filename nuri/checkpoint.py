"""Saving and loading models (ARCHITECTURE.md §15).

Two kinds of files are kept apart:
- Public weights: config.json + model.safetensors (bf16 by default). These
  are what gets released.
- Training state: optimizer, step and RNG state for resuming. Never released.

The token embedding is one tensor used under two names (`embedding.weight`
and `encoder.embedding.weight`). safetensors refuses shared tensors, so only
the first name of each tensor is written, and the aliases are filled back in
on load because they point at the same module.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nuri.config import NuriConfig
from nuri.model import NuriModel


def _unique_tensors(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Split the state dict into first-seen names and alias names of shared tensors."""
    unique, aliases, seen = {}, set(), set()
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tensor.shape, tensor.dtype)
        if key in seen:
            aliases.add(name)
        else:
            seen.add(key)
            unique[name] = tensor
    return unique, aliases


def save_pretrained(model: NuriModel, folder: str | Path, dtype: torch.dtype = torch.bfloat16) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.config.save(folder / "config.json")
    tensors, _ = _unique_tensors(model)
    weights = {name: t.detach().to("cpu", dtype).contiguous() for name, t in tensors.items()}
    save_file(weights, str(folder / "model.safetensors"), metadata={"format": "pt"})


def load_pretrained(folder: str | Path, device="cpu", dtype: torch.dtype = torch.float32) -> NuriModel:
    folder = Path(folder)
    model = NuriModel(NuriConfig.load(folder / "config.json"))
    weights = load_file(str(folder / "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    _, aliases = _unique_tensors(model)
    if unexpected or set(missing) - aliases:
        raise ValueError(f"checkpoint does not match the model: missing={missing}, unexpected={unexpected}")
    return model.to(device=device, dtype=dtype)


def save_training_state(path: str | Path, model, optimizer, step: int, extra: dict | None = None) -> None:
    """Full state for resuming a run. Kept private; never uploaded."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": model.config.to_dict(),
        "torch_rng": torch.get_rng_state(),
        "extra": extra or {},
    }, path)


def load_training_state(path: str | Path, model, optimizer) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    return state
