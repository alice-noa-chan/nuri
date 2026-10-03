"""Model configuration and the presets defined in docs/ARCHITECTURE.md §3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MoEConfig:
    num_experts: int = 16
    expert_ffn_size: int = 512
    top_k: int = 3
    num_shared_experts: int = 1
    # 1-based layer indices, matching the numbering in ARCHITECTURE.md §9.
    input_layers: tuple[int, ...] = (2, 4, 6, 8, 10)
    readout_layers: tuple[int, ...] = (2,)
    # Loss weights are calibrated from logs (§9); these are starting points.
    balance_loss_weight: float = 0.01
    z_loss_weight: float = 0.001


@dataclass
class NuriConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    num_heads: int = 8
    ffn_size: int = 1408
    input_layers: int = 6
    readout_layers: int = 2
    latent_slots: int = 16
    max_input_len: int = 512
    # Response limit including EOS; also the number of length classes.
    max_response_len: int = 64
    embed_dim: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = False
    gate_init: float = 0.1
    train_depths: tuple[int, ...] = (4,)
    default_depth: int = 4
    refine_rounds: int = 4
    num_labels: int = 0
    pad_id: int = 0
    mask_id: int = 1
    eos_id: int = 2
    initializer_range: float = 0.02
    moe: Optional[MoEConfig] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def __post_init__(self):
        if isinstance(self.moe, dict):
            self.moe = MoEConfig(**self.moe)
        if self.moe is not None:
            self.moe.input_layers = tuple(self.moe.input_layers)
            self.moe.readout_layers = tuple(self.moe.readout_layers)
        self.train_depths = tuple(self.train_depths)
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head_dim")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NuriConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NuriConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def reference_config(**overrides) -> NuriConfig:
    """49.3M structure-validation model (§3 table)."""
    return NuriConfig(**overrides)


def release_dense_config(**overrides) -> NuriConfig:
    """≈150M public dense model (D-009, D-013)."""
    base = dict(
        vocab_size=65536,
        hidden_size=768,
        num_heads=12,
        ffn_size=2048,
        input_layers=10,
        readout_layers=2,
        latent_slots=32,
        max_input_len=1024,
        max_response_len=256,
        qk_norm=True,
        train_depths=(2, 4, 8),
    )
    base.update(overrides)
    return NuriConfig(**base)


def release_moe_config(**overrides) -> NuriConfig:
    """≈242M total / ≈150M active public MoE model (D-011)."""
    overrides.setdefault("moe", MoEConfig())
    return release_dense_config(**overrides)


def tiny_config(**overrides) -> NuriConfig:
    """Small model for unit tests and CPU smoke runs."""
    base = dict(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        ffn_size=128,
        input_layers=2,
        readout_layers=2,
        latent_slots=4,
        max_input_len=32,
        max_response_len=16,
        embed_dim=32,
        qk_norm=True,
        train_depths=(1, 2, 3),
        default_depth=2,
    )
    base.update(overrides)
    return NuriConfig(**base)


PRESETS = {
    "reference": reference_config,
    "release-dense": release_dense_config,
    "release-moe": release_moe_config,
    "tiny": tiny_config,
}
