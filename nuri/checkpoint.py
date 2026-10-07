"""Saving and loading models (ARCHITECTURE.md §15).

Two kinds of files are kept apart:
- Public weights: config.json + model.safetensors (bf16 by default). These
  are what gets released.
- Training state: optimizer, step and RNG state for resuming. Never released.

Both are written atomically: the bytes go to a temporary file next to the
destination, which is flushed to disk and then renamed over it. A process
killed in the middle of a save (a preempted GPU host, for example) leaves the
previous complete file in place, never a truncated one.

The token embedding is stored once as `embedding.weight`; the vocabulary
head reuses it at run time, so no tensor is shared under two names.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .config import NuriConfig
from .model import NuriModel
from .readout import MemoryPositions
from .tokenizer import copy_tokenizer_files


def atomic_write(path: str | Path, write: Callable[[Path], None]) -> None:
    """Call write(temporary path), flush it to disk, then rename it over `path`.

    os.replace is atomic on one file system, so readers see either the old
    complete file or the new complete file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write(temporary)
        with open(temporary, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_pretrained(model: NuriModel, folder: str | Path, dtype: torch.dtype = torch.bfloat16,
                    tokenizer_dir: str | Path | None = None) -> None:
    """Write config.json + model.safetensors, plus the tokenizer files when given.

    model.safetensors is written last, so its presence means the folder is
    complete (scripts/run_release.py relies on this).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.config.save(folder / "config.json")
    if tokenizer_dir is not None:
        copy_tokenizer_files(tokenizer_dir, folder)
    weights = {name: t.detach().to("cpu", dtype).contiguous() for name, t in model.state_dict().items()}
    atomic_write(folder / "model.safetensors",
                 lambda temporary: save_file(weights, str(temporary), metadata={"format": "pt"}))


def load_pretrained(folder: str | Path, device="cpu", dtype: torch.dtype = torch.float32) -> NuriModel:
    """Load a training checkpoint or a Hugging Face release folder.

    Release folders (scripts/export_release.py) keep the config under the key
    "nuri" and prefix every weight with "model."; both layouts load here.
    """
    folder = Path(folder)
    data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = NuriModel(NuriConfig.from_dict(data["nuri"] if "nuri" in data else data))
    weights = load_file(str(folder / "model.safetensors"))
    if weights and all(name.startswith("model.") for name in weights):
        weights = {name[len("model."):]: tensor for name, tensor in weights.items()}
    model.load_state_dict(weights)  # strict: every name must match
    return model.to(device=device, dtype=dtype)


def save_training_state(path: str | Path, model, optimizer, step: int, extra: dict | None = None) -> None:
    """Full state for resuming a run. Kept private; never uploaded."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": model.config.to_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "extra": extra or {},
    }
    atomic_write(path, lambda temporary: torch.save(state, temporary))


def training_state_config(path: str | Path) -> NuriConfig:
    """The model config stored in a training state, to rebuild the model before resuming."""
    return NuriConfig.from_dict(torch.load(path, map_location="cpu", weights_only=False)["config"])


def load_training_state(path: str | Path, model, optimizer) -> dict:
    """Restore a run. Refuses a state saved with a different model config.

    Equal tensor shapes are not enough: a different rope_theta, for example,
    loads cleanly but silently changes what the model computes.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved, current = dict(state["config"]), model.config.to_dict()
    # This field did not exist in the original topology. Its only legitimate
    # resume default is the unchanged legacy behavior, never learned positions.
    saved.setdefault("canvas_position_encoding", "none")
    saved.setdefault("memory_position_encoding", "none")
    if saved != current:
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        raise ValueError(f"training state config differs from the model config in: {changed}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    return state


def add_canvas_positions(model: NuriModel, seed: int = 0) -> NuriModel:
    """Explicitly initialize a legacy model for repair training, in place.

    Every existing weight stays unchanged; only the new position table is
    created. This is a warm start requiring a fresh optimizer and schedule.
    An old training state cannot resume this changed configuration.
    """
    if model.config.canvas_position_encoding != "none" or model.readout.position_embedding is not None:
        raise ValueError("Position repair requires a legacy model without canvas positions.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("The position initialization seed must be an integer.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(model.config.max_response_len, model.config.hidden_size, generator=generator)
    weights = weights * model.config.initializer_range
    weights = weights.to(device=model.embedding.weight.device, dtype=model.embedding.weight.dtype)
    model.readout.position_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
    model.config.canvas_position_encoding = "learned"
    return model


def add_memory_positions(model: NuriModel) -> NuriModel:
    """Enable readout memory positions on a model that lacks them, in place.

    The new tables start at zero, so the model computes exactly what it did
    before; every existing weight stays unchanged. Train with a fresh
    optimizer and schedule (a warm start). An old training state cannot
    resume this changed configuration.
    """
    if model.config.memory_position_encoding != "none" or model.readout.memory_positions is not None:
        raise ValueError("This model already has readout memory positions.")
    tables = MemoryPositions(model.config.max_input_len, model.config.hidden_size)
    weight = model.embedding.weight
    model.readout.memory_positions = tables.to(device=weight.device, dtype=weight.dtype)
    model.config.memory_position_encoding = "learned"
    return model
