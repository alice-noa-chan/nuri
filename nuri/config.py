"""Model configuration and the presets defined in docs/ARCHITECTURE.md §3."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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
    # Experimental (§9, OFF): MoE FFN in the shared reasoning core. The router
    # can pick different experts at different steps although the weights are shared.
    core: bool = False
    # How expert matmuls run: "loop" (one matmul per expert, the tested default)
    # or "grouped" (one torch._grouped_mm per projection, no host sync; falls
    # back to the loop when the op or the tensor alignment is not available).
    backend: str = "loop"
    # Loss weights are calibrated from logs (§9); these are starting points.
    balance_loss_weight: float = 0.01
    z_loss_weight: float = 0.001


@dataclass
class NuriConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    num_heads: int = 8
    # Experimental grouped-query attention (§9, OFF = None): key/value heads per layer.
    num_kv_heads: int | None = None
    # Experimental MLA-style latent K/V bottleneck (§9, OFF = None): size of the shared latent.
    kv_latent_dim: int | None = None
    ffn_size: int = 1408
    input_layers: int = 6
    readout_layers: int = 2
    latent_slots: int = 16
    max_input_len: int = 512
    # Response limit including EOS; also the number of length classes.
    max_response_len: int = 64
    # Legacy configs omit this field and retain "none". New presets use
    # "learned" so identical MASK tokens can represent different positions.
    canvas_position_encoding: str = "none"
    # Experimental (OFF = "none"): "learned" adds start- and end-relative
    # position vectors to X inside the readout memory only, so a canvas
    # position can read "input token i" in order. The encoder, MLM and
    # embeddings are unchanged. The tables start at zero, so enabling it on a
    # trained model keeps its outputs until training moves them.
    memory_position_encoding: str = "none"
    embed_dim: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = False
    gate_init: float = 0.1
    train_depths: tuple[int, ...] = (4,)
    default_depth: int = 4
    refine_rounds: int = 4
    # Set to True after instruction tuning. Base models never train the length
    # head, so generation then needs an explicit canvas length (§7).
    length_head_trained: bool = False
    num_labels: int = 0
    pad_id: int = 0
    mask_id: int = 1
    eos_id: int = 2
    initializer_range: float = 0.02
    # Bidirectional local/global attention in the input encoder (§4, OFF by
    # default). One letter per input layer: "G" = global, "L" = local, e.g.
    # "GLLGLG". Local layers see keys with |i - j| <= local_radius.
    attention_schedule: str = ""
    local_radius: int = 64
    # Experimental hypothesis slot groups (§9 idea 2, OFF = 1): the K slots form
    # this many groups that attend only within their group in every core step
    # except the last one, where all slots merge.
    slot_groups: int = 1
    # Experimental draft feedback (§9, OFF): the core also reads the current
    # (corrupted) answer canvas, and generation recomputes R every round. This
    # deliberately changes the v0 contract that R never sees the canvas.
    draft_feedback: bool = False
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
        self.validate()

    def validate(self) -> None:
        """Fail early with a clear message instead of deep inside a forward pass."""
        integer_fields = (
            "vocab_size",
            "hidden_size",
            "num_heads",
            "ffn_size",
            "input_layers",
            "readout_layers",
            "latent_slots",
            "max_input_len",
            "max_response_len",
            "embed_dim",
            "default_depth",
            "refine_rounds",
            "num_labels",
            "pad_id",
            "mask_id",
            "eos_id",
            "local_radius",
            "slot_groups",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("num_kv_heads", "kv_latent_dim"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer when supplied")
        if any(type(depth) is not int for depth in self.train_depths):
            raise ValueError("reasoning depths must be integers")
        for name in ("rope_theta", "norm_eps", "gate_init", "initializer_range"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        sizes = (
            "vocab_size",
            "hidden_size",
            "num_heads",
            "ffn_size",
            "input_layers",
            "readout_layers",
            "latent_slots",
            "max_input_len",
            "max_response_len",
            "embed_dim",
        )
        for name in sizes:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_kv_heads is not None and (
            self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads
        ):
            raise ValueError("num_kv_heads must be >= 1 and divide num_heads")
        if (
            self.kv_latent_dim is not None
            and not 1 <= self.kv_latent_dim <= self.hidden_size
        ):
            raise ValueError("kv_latent_dim must be between 1 and hidden_size")
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head_dim")
        if (
            not self.train_depths
            or min(self.train_depths) < 0
            or self.default_depth < 0
        ):
            raise ValueError(
                "reasoning depths must be >= 0 and train_depths must not be empty"
            )
        if self.refine_rounds <= 0:
            raise ValueError("refine_rounds must be positive")
        if self.canvas_position_encoding not in ("none", "learned"):
            raise ValueError('canvas_position_encoding must be "none" or "learned"')
        if self.memory_position_encoding not in ("none", "learned"):
            raise ValueError('memory_position_encoding must be "none" or "learned"')
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError("gate_init is a sigmoid output and must be in (0, 1)")
        if self.norm_eps <= 0 or self.rope_theta <= 0 or self.initializer_range <= 0:
            raise ValueError(
                "norm_eps, rope_theta and initializer_range must be positive"
            )
        if self.attention_schedule:
            if len(self.attention_schedule) != self.input_layers or set(
                self.attention_schedule
            ) - {"G", "L"}:
                raise ValueError(
                    "attention_schedule needs one 'G' or 'L' per input layer"
                )
            if self.local_radius < 1:
                raise ValueError("local_radius must be >= 1")
        if self.slot_groups < 1 or self.latent_slots % self.slot_groups:
            raise ValueError("slot_groups must be >= 1 and divide latent_slots")
        if self.num_labels < 0:
            raise ValueError("num_labels must be >= 0")
        special = (self.pad_id, self.mask_id, self.eos_id)
        if len(set(special)) != 3 or not all(0 <= i < self.vocab_size for i in special):
            raise ValueError(
                "pad_id, mask_id and eos_id must be distinct ids inside the vocabulary"
            )
        if self.moe is not None:
            moe = self.moe
            for name in (
                "num_experts",
                "expert_ffn_size",
                "top_k",
                "num_shared_experts",
            ):
                if type(getattr(moe, name)) is not int:
                    raise ValueError(f"MoE {name} must be an integer")
            if any(
                type(index) is not int
                for index in (*moe.input_layers, *moe.readout_layers)
            ):
                raise ValueError("MoE layer indices must be integers")
            for name in ("balance_loss_weight", "z_loss_weight"):
                value = getattr(moe, name)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"MoE {name} must be a finite nonnegative number")
            if (
                moe.num_experts <= 0
                or moe.expert_ffn_size <= 0
                or moe.num_shared_experts < 0
            ):
                raise ValueError("MoE sizes must be positive")
            if not 2 <= moe.top_k <= moe.num_experts:
                raise ValueError("MoE top_k must be between 2 and num_experts")
            if moe.backend not in ("loop", "grouped"):
                raise ValueError('MoE backend must be "loop" or "grouped"')
            if not all(1 <= i <= self.input_layers for i in moe.input_layers):
                raise ValueError(
                    "MoE input_layers must be 1-based indices of input layers"
                )
            if not all(1 <= i <= self.readout_layers for i in moe.readout_layers):
                raise ValueError(
                    "MoE readout_layers must be 1-based indices of readout layers"
                )

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
    overrides.setdefault("canvas_position_encoding", "learned")
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
        canvas_position_encoding="learned",
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
        canvas_position_encoding="learned",
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
