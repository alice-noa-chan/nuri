"""Saving and loading models (ARCHITECTURE.md §15).

Two kinds of files are kept apart:
- Public weights: config.json + model.safetensors (bf16 by default). These
  are what gets released.
- Training state: optimizer, step and RNG state for resuming. Never released.

The token embedding is stored once as `embedding.weight`; the vocabulary
head reuses it at run time, so no tensor is shared under two names.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import NuriConfig
from .model import NuriModel
from .tokenizer import copy_tokenizer_files


def save_pretrained(model: NuriModel, folder: str | Path, dtype: torch.dtype = torch.bfloat16,
                    tokenizer_dir: str | Path | None = None) -> None:
    """Write config.json + model.safetensors, plus the tokenizer files when given."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.config.save(folder / "config.json")
    weights = {name: t.detach().to("cpu", dtype).contiguous() for name, t in model.state_dict().items()}
    save_file(weights, str(folder / "model.safetensors"), metadata={"format": "pt"})
    if tokenizer_dir is not None:
        copy_tokenizer_files(tokenizer_dir, folder)


def load_pretrained(folder: str | Path, device="cpu", dtype: torch.dtype = torch.float32) -> NuriModel:
    folder = Path(folder)
    model = NuriModel(NuriConfig.load(folder / "config.json"))
    model.load_state_dict(load_file(str(folder / "model.safetensors")))  # strict: every name must match
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
        "python_rng": random.getstate(),
        "extra": extra or {},
    }, path)


def load_training_state(path: str | Path, model, optimizer) -> dict:
    """Restore a run. Refuses a state saved with a different model config.

    Equal tensor shapes are not enough: a different rope_theta, for example,
    loads cleanly but silently changes what the model computes.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved, current = state["config"], model.config.to_dict()
    if saved != current:
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        raise ValueError(f"training state config differs from the model config in: {changed}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    return state
