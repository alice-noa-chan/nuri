"""Hugging Face config for Nuri (ARCHITECTURE.md §15).

The released config.json stores every NuriConfig field under the key "nuri",
next to the usual Hugging Face fields (model_type, auto_map, ...). This file
is copied into each model repository and loaded with trust_remote_code.
"""

from __future__ import annotations

from transformers import PretrainedConfig

from .config import NuriConfig


class NuriHFConfig(PretrainedConfig):
    model_type = "nuri"

    def __init__(self, nuri: dict | None = None, **kwargs):
        self.nuri = dict(nuri) if nuri is not None else NuriConfig().to_dict()
        kwargs.setdefault("pad_token_id", self.nuri.get("pad_id", 0))
        kwargs.setdefault("eos_token_id", self.nuri.get("eos_id", 2))
        super().__init__(**kwargs)

    @classmethod
    def from_nuri_config(cls, config: NuriConfig, **kwargs) -> "NuriHFConfig":
        return cls(nuri=config.to_dict(), **kwargs)

    def to_nuri_config(self, num_labels: int = 0) -> NuriConfig:
        """The plain NuriConfig, with a task head of `num_labels` outputs (0 = none)."""
        return NuriConfig.from_dict(dict(self.nuri, num_labels=num_labels))
