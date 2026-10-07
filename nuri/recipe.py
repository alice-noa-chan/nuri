"""Versioned, opt-in controls for existing training objectives and sampling."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


STAGE_TASKS = {"base": ("mlm", "span"), "it": ("instruction", "reasoning", "span"), "embed": ("pairs",)}
STAGE_LOSSES = {"base": set(), "it": {"length_weight"},
                "embed": {"embed_dims", "temperature", "cross_mode_weight"}}
STAGE_CORRUPTION = {"base": {"mlm_rate", "min_span", "full_mask_prob", "span_end_prob",
                             "r_dropout_prob"},
                    "it": {"min_span", "full_mask_prob", "span_end_prob"}, "embed": set()}


def _number(value, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"recipe {name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"recipe {name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"recipe {name} must be a finite number")
    if number < 0 or (positive and number == 0):
        raise ValueError(f"recipe {name} must be {'positive' if positive else 'nonnegative'}")
    return number


def _section(data: dict, name: str, allowed: set[str]) -> dict:
    section = data.get(name, {})
    if not isinstance(section, dict) or set(section) - allowed:
        raise ValueError(f"recipe {name} contains unsupported fields for this stage")
    return section


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate recipe field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TrainingRecipe:
    """A validated immutable recipe; use from_dict or load_recipe_config to construct it."""

    stage: str
    task_mix: Mapping[str, float] | None
    loss: Mapping
    corruption: Mapping

    @classmethod
    def from_dict(cls, data: dict, stage: str) -> "TrainingRecipe":
        allowed = {"schema_version", "stage", "task_mix", "loss", "corruption"}
        if (not isinstance(data, dict) or set(data) - allowed
                or type(data.get("schema_version")) is not int or data["schema_version"] != 1):
            raise ValueError("recipe needs schema_version 1 and supported fields")
        if stage not in STAGE_TASKS or data.get("stage") != stage:
            raise ValueError("recipe stage must match the requested training stage")
        mix = None
        if "task_mix" in data:
            raw = _section(data, "task_mix", set(STAGE_TASKS[stage]))
            if not raw:
                raise ValueError("recipe task_mix must be nonempty")
            # Fixed stage order avoids changing sampling when JSON keys are reordered.
            mix = {name: _number(raw[name], f"task_mix.{name}", positive=True)
                   for name in STAGE_TASKS[stage] if name in raw}
            if not math.isfinite(sum(mix.values())):
                raise ValueError("recipe task_mix total must be finite")
        loss = {}
        for name, value in _section(data, "loss", STAGE_LOSSES[stage]).items():
            if name == "embed_dims":
                if (not isinstance(value, list) or not 1 <= len(value) <= 8
                        or any(type(d) is not int or d <= 0 for d in value)
                        or value != sorted(set(value))):
                    raise ValueError("recipe embed_dims needs 1..8 strictly increasing positive integers")
                loss[name] = tuple(value)
            else:
                loss[name] = _number(value, name, positive=name == "temperature")
        corruption = {}
        for name, value in _section(data, "corruption", STAGE_CORRUPTION[stage]).items():
            if name == "min_span":
                if type(value) is not int or value < 1:
                    raise ValueError("recipe min_span must be a positive integer")
                corruption[name] = value
            else:
                corruption[name] = _number(value, name, positive=name == "mlm_rate")
                if corruption[name] > 1:
                    raise ValueError(f"recipe {name} must be at most 1")
        return cls(stage, MappingProxyType(mix) if mix is not None else None,
                   MappingProxyType(loss), MappingProxyType(corruption))

    def to_dict(self) -> dict:
        """A path-free JSON snapshot of the declared controls, including explicit defaults."""
        result = {"schema_version": 1, "stage": self.stage,
                  "loss": {k: list(v) if isinstance(v, tuple) else v for k, v in self.loss.items()},
                  "corruption": dict(self.corruption)}
        if self.task_mix is not None:
            result["task_mix"] = dict(self.task_mix)
        return result

    @property
    def sha256(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_model(self, config) -> None:
        if self.corruption.get("min_span", 1) > config.max_response_len:
            raise ValueError("recipe min_span exceeds the model response limit")
        if self.loss.get("embed_dims", (1,))[-1] > config.embed_dim:
            raise ValueError("recipe embed_dims exceeds the model embedding dimension")

    def validate_tasks(self, available) -> None:
        missing = set(self.task_mix or {}) - set(available)
        if missing:
            raise ValueError(f"recipe task_mix requests unavailable tasks: {sorted(missing)}")

    def resolve_mix(self, default: dict[str, float]) -> dict[str, float]:
        self.validate_tasks(default)
        return dict(self.task_mix) if self.task_mix is not None else default

    def bind_tasks(self, tasks) -> None:
        for task in tasks.values():
            task.options = {**task.options, "recipe": self.to_dict(), "recipe_sha256": self.sha256}


def load_recipe_config(path: str | Path | None, stage: str) -> TrainingRecipe | None:
    """Load a private recipe without opening data, model tensors or outputs."""
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    return TrainingRecipe.from_dict(data, stage)


def recipe_loss_settings(recipe: TrainingRecipe | None, cross_mode_weight: float = 0.0,
                         cross_mode_explicit: bool = False) -> dict:
    """Resolve existing loss controls; conflicting explicit CLI settings are an error."""
    overrides = dict(recipe.loss) if recipe is not None else {}
    if recipe is not None:
        cross_mode_weight = _number(cross_mode_weight, "cross_mode_weight")
    if (cross_mode_explicit and "cross_mode_weight" in overrides
            and overrides["cross_mode_weight"] != cross_mode_weight):
        raise ValueError("--cross-mode-weight conflicts with the recipe")
    overrides.setdefault("cross_mode_weight", cross_mode_weight)
    return overrides


def validate_recipe_resume(recipe: TrainingRecipe, extra: dict) -> None:
    """Recipe-enabled resumes require exact evidence, including in old checkpoints."""
    options = extra.get("task_options")
    if not isinstance(options, dict) or not options:
        raise ValueError("recipe resume requires saved task options; use --init for a new recipe run")
    for task in options.values():
        if (not isinstance(task, dict) or task.get("recipe") != recipe.to_dict()
                or task.get("recipe_sha256") != recipe.sha256):
            raise ValueError("resume requires the same recipe; use --init and a fresh output directory to change it")
