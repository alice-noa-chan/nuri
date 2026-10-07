"""Training loop and the loss for each kind of batch (ARCHITECTURE.md §8-C, §11).

A training run mixes several batch "tasks" (for example MLM and span
denoising in the base stage). Each optimizer step runs grad_accum
micro-batches; each micro-batch picks a task by its mix weight and a
reasoning depth, builds a batch and adds the gradient of its loss. Then the
optimizer steps once, the step is logged, and checkpoints are saved when due.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

from .checkpoint import (
    atomic_write,
    load_training_state,
    save_pretrained,
    save_training_state,
)
from .config import NuriConfig
from .losses import (
    contrastive_loss,
    length_loss,
    validate_vocab_loss,
    vocab_sample_mean_ce,
    vocab_token_mean_ce,
)
from .model import NuriModel
from .monitoring import TrainingMonitor
from .schedule import WSDSchedule

# ----- loss for each batch kind ------------------------------------------------------


def mlm_loss(
    model: NuriModel,
    batch: dict,
    *,
    vocab_loss_backend: str = "torch",
    vocab_chunk_size: int = 256,
) -> dict:
    """Input MLM: predicts masked input tokens straight from X (no core, no readout)."""
    x, _ = model.encode(batch["input_ids"], batch["input_valid"], batch.get("segments"))
    mask = batch["target_mask"]
    loss = vocab_token_mean_ce(
        x[mask],
        model.embedding.weight,
        batch["labels"][mask],
        vocab_loss_backend,
        vocab_chunk_size,
    )
    return {
        "loss": loss,
        "tokens": int(batch["input_valid"].sum()),
        "targets": int(mask.sum()),
    }


def canvas_loss(
    model: NuriModel,
    batch: dict,
    depth: int,
    length_weight: float = 0.0,
    *,
    vocab_loss_backend: str = "torch",
    vocab_chunk_size: int = 256,
) -> dict:
    """Span denoising and instruction tuning: fill the masked canvas from [X; R].

    The length loss is added only for batches with natural answer lengths
    (instruction tuning). Random span lengths must never train it (§7).
    """
    x, valid = model.encode(batch["input_ids"], batch["input_valid"])
    draft = (
        (batch["canvas"], batch["canvas_valid"])
        if model.config.draft_feedback
        else None
    )
    r = model.reason(x, valid, depth, draft=draft)
    cache = model.generation_memory(x, valid, r)
    if model.training and "hide_r" in batch:
        # R dropout (recipe r_dropout_prob): these samples read only X, so the
        # readout cannot rely on the 32-slot R summary alone. Evaluation and
        # generation always see the full memory.
        cache.mask[batch["hide_r"], x.shape[1] :] = False
    hidden = model.readout_hidden(batch["canvas"], batch["canvas_valid"], cache)
    mask = batch["loss_mask"]
    loss = vocab_sample_mean_ce(
        hidden[mask],
        model.embedding.weight,
        batch["targets"],
        mask,
        vocab_loss_backend,
        vocab_chunk_size,
    )
    result = {"denoise": loss.detach()}
    if length_weight:
        # The training canvas has the gold length, so the length head must use an
        # R that has not seen the draft.
        r_length = model.reason(x, valid, depth) if draft is not None else r
        u, r_mean = model.pool(x, valid, r_length)
        length = length_loss(model.length_logits(u, r_mean), batch["lengths"])
        loss = loss + length_weight * length
        result["length"] = length.detach()
    tokens = int(batch["input_valid"].sum() + batch["canvas_valid"].sum())
    return dict(result, loss=loss, tokens=tokens, targets=int(mask.sum()))


def pair_loss(
    model: NuriModel,
    batch: dict,
    depth: int,
    dims: tuple[int, ...],
    temperature: float,
    cross_mode_weight: float = 0.0,
) -> dict:
    """Contrastive embedding loss, trained for both the fast and the reasoning mode.

    Query and document use the same mode, as the spec's default retrieval
    does (§6). Each side is encoded once and its X is reused for both modes,
    so the encoder runs twice per step, not four times.

    cross_mode_weight > 0 adds the cross-mode alignment of §6: reasoning-mode
    queries against fast-mode documents, so documents can be indexed cheaply
    with the fast mode while queries think. Off by default.
    """
    xq, vq = model.encode(batch["query_ids"], batch["query_valid"])
    xd, vd = model.encode(batch["doc_ids"], batch["doc_valid"])
    embeddings, losses = {}, {}
    for mode in ("fast", "reasoning"):
        embeddings[mode] = (
            model.embed_memory(xq, vq, mode=mode, steps=depth),
            model.embed_memory(xd, vd, mode=mode, steps=depth),
        )
        losses[mode] = contrastive_loss(
            *embeddings[mode], temperature, dims, batch.get("false_negatives")
        )
    loss = (losses["fast"] + losses["reasoning"]) / 2
    result = {
        "fast": losses["fast"].detach(),
        "reasoning": losses["reasoning"].detach(),
    }
    if cross_mode_weight:
        cross = contrastive_loss(
            embeddings["reasoning"][0],
            embeddings["fast"][1],
            temperature,
            dims,
            batch.get("false_negatives"),
        )
        loss = loss + cross_mode_weight * cross
        result["cross_mode"] = cross.detach()
    tokens = int(batch["query_valid"].sum() + batch["doc_valid"].sum())
    return dict(result, loss=loss, tokens=tokens, targets=batch["query_ids"].shape[0])


# ----- training loop -----------------------------------------------------------------


@dataclass
class TrainSettings:
    out_dir: str
    total_steps: int
    peak_lr: float = 3e-4
    warmup_steps: int = 1000
    decay_fraction: float = 0.15
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 0  # 0 = only pre-decay and final checkpoints
    seed: int = 0
    length_weight: float = 0.1  # used by "instruction" batches
    embed_dims: tuple[int, ...] = (64, 128, 256)
    temperature: float = 0.05
    cross_mode_weight: float = (
        0.0  # embed stage: align reasoning queries with fast documents (§6)
    )
    mix: dict[str, float] = field(default_factory=dict)
    tokenizer_dir: str | None = None  # copied into every saved checkpoint
    stage: str = "training"  # label only; the task mix defines the computation
    progress: bool = True
    tensorboard_dir: str | None = None  # None = no event files or TensorBoard import
    grad_accum: int = (
        1  # micro-batches per optimizer step (each draws its own task and depth)
    )
    gradient_checkpointing: bool = (
        False  # recompute block activations in backward (same results, less memory)
    )
    compile: bool = False  # torch.compile the blocks in place (same weights and names)
    native_gqa: bool = (
        False  # let SDPA consume compact grouped K/V heads (runtime option)
    )
    attention_backend: str = (
        "sdpa"  # optional external FlashAttention kernel; arbitrary masks use SDPA
    )
    liger_kernels: bool = False  # optional fused RMSNorm and dense SwiGLU
    vocab_loss_backend: str = (
        "torch"  # torch, checkpointed chunks, or optional Liger loss
    )
    vocab_chunk_size: int = (
        256  # selected tokens per checkpointed vocabulary projection
    )
    save_every_minutes: float = (
        0.0  # also save training_state/latest.pt this often (0 = off)
    )
    valid_every: int = (
        0  # completed optimizer steps between bounded validations (0 = final only)
    )
    stop_after_completed_steps: int | None = (
        None  # runtime gate; preserves the full optimizer schedule
    )


@dataclass
class Task:
    """A named batch source. kind is "mlm", "span", "instruction" or "pairs".

    rng and generator are the random sources next_batch draws from. The
    trainer saves and restores them, so a resumed run sees the same batches
    as an uninterrupted one.

    options records how batches are built (batch size, pool sizes, ...). It
    is saved with the training state and must match on resume, because a
    different batch size would silently change the data of a resumed run.
    """

    kind: str
    next_batch: Callable[[], dict]
    rng: random.Random | None = None
    generator: torch.Generator | None = None
    options: dict = field(default_factory=dict)


@dataclass
class HoldoutSet:
    """Explicit held-out tasks sampled with the same corruption on each assessment.

    Sources follow Task's RNG protocol and must not hide mutable iterator state.
    Only one batch is held in memory. Sampling is bounded, not a full traversal
    of the source pool; repeated rows are presentations, not unique coverage.
    """

    tasks: dict[str, Task]
    batches_per_task: int = 8
    seed: int = 0
    depth: int = 4
    metadata: dict = field(default_factory=dict)
    _initial_random: dict = field(init=False, repr=False)

    def __post_init__(self):
        if not self.tasks or self.batches_per_task < 1 or self.depth < 0:
            raise ValueError(
                "holdout needs tasks, positive batches_per_task and nonnegative depth"
            )
        if any(
            task.kind not in {"mlm", "span", "instruction", "pairs"}
            for task in self.tasks.values()
        ):
            raise ValueError("unknown holdout task kind")
        self._initial_random = _task_random_state(self.tasks)


class HoldoutInterrupted(Exception):
    """A stop arrived between bounded assessment batches; training handles the save."""


def _holdout_policy_sha256(holdout: HoldoutSet) -> str:
    sources = {}
    for name, state in holdout._initial_random.items():
        generator = state["generator"]
        sources[name] = {
            "rng": state["rng"],
            "generator_sha256": None
            if generator is None
            else hashlib.sha256(generator.numpy().tobytes()).hexdigest(),
        }
    policy = {
        "seed": holdout.seed,
        "depth": holdout.depth,
        "batches_per_task": holdout.batches_per_task,
        "metadata": holdout.metadata,
        "task_options": _task_options(holdout.tasks),
        "random": sources,
    }
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _model_assessment_sha256(model: NuriModel, chunk_bytes: int = 1024 * 1024) -> str:
    """Bind exact evaluated weights and config with bounded host transfer memory."""
    digest = hashlib.sha256(
        json.dumps(model.config.to_dict(), sort_keys=True, allow_nan=False).encode()
    )

    def chunks(tensor):
        count = max(1, chunk_bytes // tensor.element_size())
        if tensor.is_contiguous():
            flat = tensor.reshape(-1)
            for start in range(0, flat.numel(), count):
                yield flat[start : start + count]
        elif tensor.numel() <= count:
            yield tensor.contiguous()
        else:
            for row in tensor:
                yield from chunks(row)

    for name, tensor in sorted(model.state_dict().items()):
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        for chunk in chunks(tensor.detach()):
            digest.update(
                chunk.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


def _objective_settings(settings: TrainSettings) -> dict:
    result = {
        "length_weight": settings.length_weight,
        "embed_dims": list(settings.embed_dims),
        "temperature": settings.temperature,
        "cross_mode_weight": settings.cross_mode_weight,
    }
    # Preserve old default receipts while binding opt-in numerical execution.
    if (
        settings.attention_backend != "sdpa"
        or settings.liger_kernels
        or settings.vocab_loss_backend != "torch"
        or settings.vocab_chunk_size != 256
    ):
        result["runtime_kernels"] = {
            key: getattr(settings, key) for key in KERNEL_SETTING_DEFAULTS
        }
    return result


def _public_checkpoint_matches(
    model: NuriModel, folder: Path, chunk_bytes: int = 1024 * 1024
) -> bool:
    """Compare a final export against the resumed state in its stored precision."""
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return False
        config = NuriConfig.from_dict(config.get("nuri", config))
    except (OSError, ValueError, TypeError):
        return False
    if config.to_dict() != model.config.to_dict():
        return False

    def selectors(shape, size, prefix=()):
        if not shape:
            yield prefix
            return
        row_bytes = math.prod(shape[1:]) * size
        if row_bytes > chunk_bytes:
            for index in range(shape[0]):
                yield from selectors(shape[1:], size, (*prefix, index))
        else:
            rows = max(1, chunk_bytes // max(row_bytes, 1))
            for start in range(0, shape[0], rows):
                yield (
                    *prefix,
                    slice(start, min(start + rows, shape[0])),
                    *(slice(None) for _ in shape[1:]),
                )

    state = model.state_dict()
    with safe_open(
        str(folder / "model.safetensors"), framework="pt", device="cpu"
    ) as saved:
        if set(saved.keys()) != set(state):
            return False
        for name, tensor in state.items():
            view = saved.get_slice(name)
            if view.get_shape() != list(tensor.shape):
                return False
            if not tensor.ndim:
                value = saved.get_tensor(name)
                if not torch.equal(
                    value, tensor.detach().to(device="cpu", dtype=value.dtype)
                ):
                    return False
                continue
            for index in selectors(tensor.shape, tensor.element_size()):
                value = view[index]
                if not torch.equal(
                    value, tensor[index].detach().to(device="cpu", dtype=value.dtype)
                ):
                    return False
    return True


def _complete_assessment(
    report: dict, holdout: HoldoutSet, settings: TrainSettings
) -> bool:
    """An interrupted or malformed receipt is never a completed evaluation."""
    try:
        if report["status"] != "complete" or set(report["tasks"]) != set(holdout.tasks):
            return False
        if any(
            type(report[key]) is not int
            for key in ("step", "seed", "depth", "assessed_rows", "assessed_batches")
        ):
            return False
        if (
            report["seed"] != holdout.seed
            or report["depth"] != holdout.depth
            or report["metadata"] != holdout.metadata
        ):
            return False
        rows, batches = 0, 0
        for name, task in holdout.tasks.items():
            record = report["tasks"][name]
            counts, metrics = record["counts"], record["metrics"]
            if (
                record["kind"] != task.kind
                or record["batches"] != holdout.batches_per_task
                or counts["batches"] != holdout.batches_per_task
                or counts["rows"] < 1
                or counts["targets"] < 1
            ):
                return False
            keys = (
                {"loss"}
                if task.kind == "mlm"
                else {"fast", "reasoning"}
                if task.kind == "pairs"
                else {"denoise"}
            )
            if task.kind == "pairs" and settings.cross_mode_weight:
                keys.add("cross_mode")
            if task.kind == "instruction" and settings.length_weight:
                keys.add("length")
            numerators, denominators = (
                record["loss_numerators"],
                record["loss_denominators"],
            )
            if (
                set(numerators) != keys
                or set(denominators) != keys
                or set(metrics) != keys | {"loss"}
            ):
                return False
            for key in keys:
                denominator = denominators[key]
                if (
                    type(denominator) is not int
                    or denominator < 1
                    or not math.isfinite(numerators[key])
                    or not math.isfinite(metrics[key])
                    or not math.isclose(
                        metrics[key],
                        numerators[key] / denominator,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    return False
                expected = counts["targets"] if task.kind == "mlm" else counts["rows"]
                if (key == "denoise" and denominator > expected) or (
                    key != "denoise" and denominator != expected
                ):
                    return False
            expected_loss = (
                metrics["loss"]
                if task.kind == "mlm"
                else (
                    (metrics["fast"] + metrics["reasoning"]) / 2
                    + settings.cross_mode_weight * metrics.get("cross_mode", 0.0)
                    if task.kind == "pairs"
                    else metrics["denoise"]
                    + settings.length_weight * metrics.get("length", 0.0)
                )
            )
            if not math.isfinite(metrics["loss"]) or not math.isclose(
                metrics["loss"], expected_loss, rel_tol=1e-12, abs_tol=1e-12
            ):
                return False
            for key in ("batch_sha256", "target_sha256"):
                digest = record[key]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    return False
            if task.kind == "pairs":
                candidates = record["candidate_rows_per_batch"]
                if (
                    len(candidates) != record["batches"]
                    or any(type(value) is not int or value < 2 for value in candidates)
                    or sum(candidates) != counts["rows"]
                ):
                    return False
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in counts.values()
            ):
                return False
            if (
                any(
                    type(value) is not int
                    for key, value in counts.items()
                    if key != "valid_fraction"
                )
                or counts["valid_fraction"] > 1
                or counts["valid_input_tokens"] > counts["padded_input_tokens"]
                or counts["valid_canvas_tokens"] > counts["padded_canvas_tokens"]
            ):
                return False
            rows += counts["rows"]
            batches += record["batches"]
        return (
            report["assessed_rows"] == rows
            and report["assessed_batches"] == batches
            and report["coverage"] == "bounded_sample"
            and math.isfinite(report["seconds"])
            and report["seconds"] >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _assessment_recorded(
    path: Path,
    split: str,
    step: int,
    holdout: HoldoutSet,
    model: NuriModel,
    settings: TrainSettings,
) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        return False
    if (
        not isinstance(report, dict)
        or report.get("split") != split
        or report.get("step") != step
        or report.get("policy_sha256") != _holdout_policy_sha256(holdout)
        or report.get("objective_settings") != _objective_settings(settings)
        or not _complete_assessment(report, holdout, settings)
    ):
        return False
    return report.get("model_sha256") == _model_assessment_sha256(model)


def _task_random_state(tasks: dict[str, Task]) -> dict:
    return {
        name: {
            "rng": task.rng.getstate() if task.rng is not None else None,
            "generator": task.generator.get_state()
            if task.generator is not None
            else None,
        }
        for name, task in tasks.items()
    }


def _restore_task_random(tasks: dict[str, Task], states: dict) -> None:
    for name, state in states.items():
        if state["rng"] is not None:
            tasks[name].rng.setstate(state["rng"])
        if state["generator"] is not None:
            tasks[name].generator.set_state(state["generator"])


def _digest_tensor(digest, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _holdout_digests(batch: dict, kind: str, batch_digest, target_digest) -> None:
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            _digest_tensor(batch_digest, key, value)
    if kind == "pairs":
        for key in ("query_ids", "query_valid", "doc_ids", "doc_valid"):
            _digest_tensor(target_digest, key, batch[key])
    else:
        mask_key, labels = (
            ("target_mask", "labels") if kind == "mlm" else ("loss_mask", "targets")
        )
        _digest_tensor(target_digest, mask_key, batch[mask_key])
        _digest_tensor(
            target_digest, "selected_targets", batch[labels][batch[mask_key]]
        )


def evaluate_holdout(
    model: NuriModel,
    holdout: HoldoutSet,
    settings: TrainSettings,
    device: torch.device,
    split: str,
    completed_steps: int,
    training_tasks: dict[str, Task] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict:
    """Assess fixed bounded batches without changing the next training update.

    Data objectives exclude MoE auxiliary losses. MLM is token weighted;
    canvas CE is weighted by target-bearing rows, length CE by all rows, and
    contrastive CE by pair rows in each fixed in-batch candidate pool.
    """
    if split not in {"valid", "test"}:
        raise ValueError("holdout split must be valid or test")
    training_tasks = training_tasks or {}
    random_state, torch_state = random.getstate(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    source_states = _task_random_state(holdout.tasks)
    training_states = _task_random_state(training_tasks)
    modes = [(module, module.training) for module in model.modules()]
    moe_state = [
        (
            layer,
            layer.last_aux,
            list(layer.aux_records),
            layer.last_backend,
            set(layer.used_backends),
        )
        for layer in model.moe_layers()
    ]
    started = time.perf_counter()
    report = {
        "split": split,
        "step": completed_steps,
        "coverage": "bounded_sample",
        "seed": holdout.seed,
        "depth": holdout.depth,
        "tasks": {},
        "policy_sha256": _holdout_policy_sha256(holdout),
        "metadata": holdout.metadata,
        "unassessed_training_tasks": sorted(set(training_tasks) - set(holdout.tasks)),
    }
    report["model_sha256"] = _model_assessment_sha256(model)
    report["objective_settings"] = _objective_settings(settings)
    try:
        model.eval()
        random.seed(holdout.seed)
        torch.manual_seed(holdout.seed)
        _restore_task_random(holdout.tasks, holdout._initial_random)
        with torch.no_grad():
            for name, task in holdout.tasks.items():
                numerators, denominators, shapes = {}, {}, []
                batch_digest, target_digest = hashlib.sha256(), hashlib.sha256()
                for _ in range(holdout.batches_per_task):
                    if stop_requested is not None and stop_requested():
                        raise HoldoutInterrupted()
                    batch = task.next_batch()
                    shape = _batch_shape(task.kind, batch, holdout.depth)
                    shapes.append(shape)
                    _holdout_digests(batch, task.kind, batch_digest, target_digest)
                    if task.kind == "mlm":
                        counts = {"loss": shape["targets"]}
                    elif task.kind == "pairs":
                        if shape["rows"] < 2:
                            raise ValueError(
                                "holdout pairs need at least two in-batch candidates"
                            )
                        counts = {
                            key: shape["rows"]
                            for key in ("fast", "reasoning", "cross_mode")
                        }
                    else:
                        counts = {
                            "denoise": int(batch["loss_mask"].any(dim=1).sum()),
                            "length": shape["rows"],
                        }
                    if min(counts.values()) < 1:
                        raise ValueError("holdout batch has no loss-bearing targets")
                    batch = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        result = _task_loss(
                            model, task.kind, batch, holdout.depth, settings
                        )
                    if not bool(torch.isfinite(result["loss"])):
                        raise ValueError(
                            "non-finite held-out objective; final weights were not promoted"
                        )
                    for key, count in counts.items():
                        if key not in result:
                            continue
                        value = float(result[key].detach())
                        if not torch.isfinite(torch.tensor(value)):
                            raise ValueError(
                                "non-finite held-out objective; final weights were not promoted"
                            )
                        numerators[key] = numerators.get(key, 0.0) + value * count
                        denominators[key] = denominators.get(key, 0) + count
                metrics = {
                    key: numerators[key] / denominators[key] for key in numerators
                }
                if task.kind in {"span", "instruction"}:
                    metrics["loss"] = metrics[
                        "denoise"
                    ] + settings.length_weight * metrics.get("length", 0.0)
                elif task.kind == "pairs":
                    metrics["loss"] = (metrics["fast"] + metrics["reasoning"]) / 2
                    metrics["loss"] += settings.cross_mode_weight * metrics.get(
                        "cross_mode", 0.0
                    )
                task_report = {
                    "kind": task.kind,
                    "metrics": metrics,
                    "loss_numerators": numerators,
                    "loss_denominators": denominators,
                    "counts": _summarize_batch_shapes(shapes),
                    "batches": len(shapes),
                    "batch_sha256": batch_digest.hexdigest(),
                    "target_sha256": target_digest.hexdigest(),
                }
                task_report["counts"]["batches"] = task_report["counts"].pop(
                    "microbatches"
                )
                task_report["source_pool"] = {
                    key: task.options[key]
                    for key in (
                        "corpus_documents",
                        "examples",
                        "pairs",
                        "source_rows",
                        "eligible_rows",
                        "selected_pool_rows",
                        "excluded_rows",
                        "selection",
                    )
                    if key in task.options
                }
                if task.kind == "pairs":
                    task_report["candidate_rows_per_batch"] = [
                        shape["rows"] for shape in shapes
                    ]
                    task_report["contrastive_pool"] = "within_each_fixed_batch"
                report["tasks"][name] = task_report
        if stop_requested is not None and stop_requested():
            raise HoldoutInterrupted()
    finally:
        for module, mode in modes:
            module.training = mode
        _restore_task_random(holdout.tasks, source_states)
        _restore_task_random(training_tasks, training_states)
        random.setstate(random_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for layer, last_aux, records, backend, used_backends in moe_state:
            layer.last_aux, layer.aux_records, layer.last_backend = (
                last_aux,
                records,
                backend,
            )
            layer.used_backends = used_backends
    report["seconds"] = time.perf_counter() - started
    report["assessed_batches"] = sum(
        task["batches"] for task in report["tasks"].values()
    )
    report["assessed_rows"] = sum(
        task["counts"]["rows"] for task in report["tasks"].values()
    )
    report["status"] = "complete"
    return report


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model: NuriModel, settings: TrainSettings) -> torch.optim.AdamW:
    """AdamW without weight decay on norms, gates, slots, embeddings and position tables.

    Position tables are embeddings too: canvas positions match "embedding"
    by name, while the readout memory positions (D-024) and draft positions
    are matched by "positions". Decay would keep pulling the zero-initialized
    memory tables back toward zero.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2 or any(part in name for part in ("embedding", "slots", "positions")):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": settings.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=settings.peak_lr, betas=settings.betas)


def _clipping_counts(totals: dict, start_step: int) -> dict:
    """Retain observation history, or start an honest window for legacy states."""
    return totals.setdefault(
        "optimizer_clipping",
        {
            "start_completed_step": start_step,
            "completed_updates": 0,
            "clipped_updates": 0,
        },
    )


def _gradient_parameter_groups(model: NuriModel) -> dict:
    """Disjoint parameter references; the tied vocabulary weight is counted once."""
    groups = {
        name: [] for name in ("encoder", "core", "readout", "tied_embedding", "other")
    }
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        prefix = name.split(".", 1)[0]
        group = (
            "tied_embedding"
            if prefix == "embedding"
            else prefix
            if prefix in groups
            else "other"
        )
        groups[group].append(parameter)
    return groups


def _tensor_chunks(tensor: torch.Tensor, max_elements: int):
    """Yield bounded views, including non-contiguous gradients, without flattening copies."""
    if tensor.numel() <= max_elements:
        yield tensor
        return
    dimension = next(index for index, size in enumerate(tensor.shape) if size > 1)
    elements_per_slice = tensor.numel() // tensor.shape[dimension]
    width = max(1, max_elements // elements_per_slice)
    for start in range(0, tensor.shape[dimension], width):
        part = tensor.narrow(
            dimension, start, min(width, tensor.shape[dimension] - start)
        )
        yield from _tensor_chunks(part, max_elements)


@torch.no_grad()
def _sample_optimizer_diagnostics(
    model: NuriModel, groups: dict, chunk_elements: int = 262144
) -> dict:
    """Observe combined gradients and current positions before clipping/optimizer.step."""
    device = next(model.parameters()).device
    squared = []
    for parameters in groups.values():
        total = torch.zeros((), dtype=torch.float32, device=device)
        for parameter in parameters:
            if parameter.grad is not None:
                for part in _tensor_chunks(parameter.grad.detach(), chunk_elements):
                    total += part.float().square().sum()
        squared.append(total)
    norms = torch.stack(squared).sqrt().tolist()
    result = {"pre_clip_grad_norms": dict(zip(groups, norms))}
    positions = model.readout.position_embedding
    if positions is not None:
        weight = positions.weight.detach()
        # Nuri's table has one short D-dimensional row per output position.
        # Only row norms are retained; no vocabulary or parameter snapshot is made.
        rows_per_chunk = max(1, chunk_elements // weight.shape[1])
        rows = torch.cat(
            [part.float().norm(dim=1) for part in weight.split(rows_per_chunk)]
        )
        values = torch.stack(
            (rows.min(), rows.mean(), rows.quantile(0.5), rows.max())
        ).tolist()
        result["canvas_position_row_norms"] = dict(
            zip(("min", "mean", "median", "max"), values)
        )
        result["canvas_position_row_norms"]["rows"] = weight.shape[0]
    return result


def train(
    model: NuriModel,
    tasks: dict[str, Task],
    settings: TrainSettings,
    device=None,
    resume_from: str | Path | None = None,
    validation: HoldoutSet | None = None,
    test: HoldoutSet | None = None,
) -> list[dict]:
    """Run the loop. Returns the logged records (also written to train_log.jsonl).

    resume_from: a private training-state file saved by an earlier run with
    the same settings. Training continues at the saved step with the same
    weights, optimizer moments, token totals and random state, so the result
    matches a run that was never interrupted.
    """
    if settings.log_every <= 0:
        raise ValueError("log_every must be positive")
    if settings.grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    boundary = settings.stop_after_completed_steps
    if boundary is not None and (
        type(boundary) is not int or not 0 < boundary < settings.total_steps
    ):
        raise ValueError(
            "stop_after_completed_steps must be a positive integer below total_steps"
        )
    if settings.valid_every < 0:
        raise ValueError("valid_every must be nonnegative")
    if settings.valid_every and validation is None:
        raise ValueError("valid_every requires explicit validation tasks")
    training_sources = {id(task) for task in tasks.values()}
    if any(
        id(task) in training_sources
        for holdout in (validation, test)
        if holdout is not None
        for task in holdout.tasks.values()
    ):
        raise ValueError("held-out tasks must be separate from training tasks")
    device = torch.device(device or default_device())
    validate_vocab_loss(settings.vocab_loss_backend, settings.vocab_chunk_size, device)
    model.to(device).train()
    model.set_gradient_checkpointing(settings.gradient_checkpointing)
    model.set_native_gqa(settings.native_gqa)
    model.set_attention_backend(settings.attention_backend)
    model.set_liger_kernels(settings.liger_kernels)
    if settings.compile:
        _compile_blocks(model)
    out = Path(settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_settings.json").write_text(
        json.dumps(asdict(settings), indent=2), encoding="utf-8"
    )

    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)
    optimizer = make_optimizer(model, settings)
    schedule = WSDSchedule(
        settings.peak_lr,
        settings.total_steps,
        settings.warmup_steps,
        settings.decay_fraction,
    )
    names = list(settings.mix)
    weights = [settings.mix[n] for n in names]
    use_bf16 = device.type == "cuda"
    totals = {"tokens": 0, "targets": 0}
    start_step = 0
    if resume_from is not None:
        start_step, totals = _resume(
            resume_from, model, optimizer, rng, tasks, settings
        )
        _clipping_counts(totals, start_step)
        if boundary is not None and start_step >= boundary:
            # Repeating a gated invocation must not complete due checkpoints or
            # consult holdouts. Omitting the boundary resumes the full schedule.
            path = _save_state(
                model,
                optimizer,
                settings,
                start_step - 1,
                totals,
                out,
                rng,
                tasks,
                LATEST_STATE,
            )
            raise TrainingInterrupted(start_step, path, reason="step_boundary")
        if (
            start_step == settings.total_steps
            and (validation or test)
            and (out / "final" / "model.safetensors").exists()
        ):
            try:
                matches = _public_checkpoint_matches(model, out / "final")
            except (OSError, SafetensorError):
                matches = False
            if not matches:
                raise ValueError(
                    "existing final checkpoint differs from the resumed model/config; use a fresh out directory"
                )
        # Checkpoints due at the resumed boundary that were not completed (a
        # stop skips them; a crash may land between state and weights) are
        # written now from the restored state, which is exactly what they
        # would have contained.
        if start_step > 0 and not (
            start_step == settings.total_steps and (validation or test)
        ):
            _complete_checkpoints(
                model,
                optimizer,
                schedule,
                settings,
                start_step - 1,
                totals,
                out,
                rng,
                tasks,
            )
    clipping = _clipping_counts(totals, start_step)
    gradient_groups = _gradient_parameter_groups(model)
    records, started = [], time.time()
    last_timed_save = time.monotonic()
    _reset_cuda_peaks(device)

    with (
        _StopOnSigterm(),
        TrainingMonitor(
            settings.total_steps,
            start_step,
            totals["tokens"],
            stage=settings.stage,
            progress=settings.progress,
            tensorboard_dir=settings.tensorboard_dir,
            resuming=resume_from is not None,
        ) as monitor,
        open(out / "train_log.jsonl", "a", encoding="utf-8") as log,
    ):

        def assess(
            holdout: HoldoutSet, split: str, step: int, skip_recorded: bool = False
        ):
            monitor.phase(f"assessing {split}", force=True)
            receipt = out / "holdout" / f"{split}-step{step + 1}.json"
            try:
                if _stop.requested:
                    raise HoldoutInterrupted()
                if skip_recorded and _assessment_recorded(
                    receipt, split, step + 1, holdout, model, settings
                ):
                    return False
                report = evaluate_holdout(
                    model,
                    holdout,
                    settings,
                    device,
                    split,
                    step + 1,
                    training_tasks=tasks,
                    stop_requested=lambda: _stop.requested,
                )
                if not _complete_assessment(report, holdout, settings):
                    raise ValueError(
                        "held-out assessment is incomplete; final weights were not promoted"
                    )
                with open(out / "holdout_log.jsonl", "ab+") as holdout_log:
                    holdout_log.seek(0, 2)
                    if holdout_log.tell():
                        holdout_log.seek(-1, 2)
                        if holdout_log.read(1) != b"\n":
                            holdout_log.write(
                                b"\n"
                            )  # preserve the incomplete tail without swallowing this record
                    holdout_log.write(
                        (json.dumps(report, allow_nan=False) + "\n").encode()
                    )
                    holdout_log.flush()
                monitor.record_holdout(report)
                # The atomic receipt is the completion evidence, written after
                # the log and TensorBoard flush. JSONL history alone cannot skip test.
                atomic_write(
                    receipt,
                    lambda temporary: temporary.write_text(
                        json.dumps(report, allow_nan=False, indent=2), encoding="utf-8"
                    ),
                )
                return True
            except HoldoutInterrupted:
                monitor.phase("saving latest state", force=True)
                path = _save_state(
                    model,
                    optimizer,
                    settings,
                    step,
                    totals,
                    out,
                    rng,
                    tasks,
                    LATEST_STATE,
                )
                raise TrainingInterrupted(step + 1, path) from None
            except Exception:
                # Assessment cannot corrupt optimizer state. Keep its last
                # completed update recoverable, but publish no final weights.
                _save_state(
                    model,
                    optimizer,
                    settings,
                    step,
                    totals,
                    out,
                    rng,
                    tasks,
                    LATEST_STATE,
                )
                raise

        if start_step == settings.total_steps and (validation or test):
            # An interrupted final assessment already completed its last update.
            # It must finish before recovery publishes the final public marker.
            reassessed = False
            if validation is not None:
                reassessed |= assess(
                    validation, "valid", start_step - 1, skip_recorded=True
                )
            if test is not None:
                reassessed |= assess(test, "test", start_step - 1, skip_recorded=True)
            if _stop.requested:
                path = _save_state(
                    model,
                    optimizer,
                    settings,
                    start_step - 1,
                    totals,
                    out,
                    rng,
                    tasks,
                    LATEST_STATE,
                )
                raise TrainingInterrupted(start_step, path)
            if reassessed:
                _maybe_save(
                    model,
                    optimizer,
                    schedule,
                    settings,
                    start_step - 1,
                    totals,
                    out,
                    rng,
                    tasks,
                )
            else:
                _complete_checkpoints(
                    model,
                    optimizer,
                    schedule,
                    settings,
                    start_step - 1,
                    totals,
                    out,
                    rng,
                    tasks,
                )
        for step in range(start_step, settings.total_steps):
            for group in optimizer.param_groups:
                group["lr"] = schedule.lr(step)
            schedule_phase = (
                "warmup"
                if step < schedule.warmup_steps
                else ("stable" if step < schedule.decay_start else "decay")
            )
            optimizer.zero_grad(set_to_none=True)
            micro_batches = []
            # With grad_accum > 1 the gradients of several micro-batches add up
            # before one optimizer step; each loss is divided so the step uses
            # their mean. grad_accum = 1 is exactly the plain loop.
            for micro in range(settings.grad_accum):
                name = rng.choices(names, weights)[0]
                depth = rng.choice(model.config.train_depths)
                monitor.begin_step(
                    name,
                    depth if tasks[name].kind != "mlm" else 0,
                    schedule_phase,
                    micro=micro + 1,
                    micro_count=settings.grad_accum,
                )
                batch = tasks[name].next_batch()
                shape = _batch_shape(tasks[name].kind, batch, depth)
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                monitor.phase("forward")
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    result = _task_loss(model, tasks[name].kind, batch, depth, settings)
                    result["batch_shape"] = shape
                    aux = model.pop_moe_aux()
                    loss = result["loss"] + (
                        aux["loss"] if aux["loss"] is not None else 0.0
                    )

                monitor.phase("backward")
                (loss / settings.grad_accum).backward()
                # Accumulated summaries log the depth the core actually ran (0 for MLM,
                # which never runs the core); grad_accum = 1 keeps the sampled depth.
                active = (
                    depth
                    if settings.grad_accum == 1 or tasks[name].kind != "mlm"
                    else 0
                )
                micro_batches.append((name, active, result, aux, loss.detach()))
                totals["tokens"] += result["tokens"]
                totals["targets"] += result["targets"]
            sampled_record = (
                step % settings.log_every == 0 or step == settings.total_steps - 1
            )
            name, depth, result, aux, loss = _merge_micro_batches(
                micro_batches, tasks, sampled_record
            )

            monitor.phase("optimizer")
            diagnostics = (
                _sample_optimizer_diagnostics(model, gradient_groups)
                if sampled_record
                else {}
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.grad_clip
            )
            # One guard/count host sync per update: a NaN/inf loss (including the MoE aux
            # terms) gives a non-finite gradient norm, and either one stops the
            # run before optimizer.step, so neither the weights nor any saved
            # state (latest.pt, step-*, final) can contain it. Recovery resumes
            # from the previous complete state.
            losses = torch.stack([micro[4].float() for micro in micro_batches])
            finite = torch.isfinite(grad_norm) & torch.isfinite(losses).all()
            # Match clip_grad_norm_'s coefficient, including its epsilon. This
            # measures gradient shrinkage, not an effective Adam learning rate.
            clip_applied = settings.grad_clip / (grad_norm + 1e-6) < 1.0
            is_finite, was_clipped = torch.stack((finite, clip_applied)).tolist()
            if not is_finite:
                raise NonFiniteTraining(step, name)
            optimizer.step()
            clipping["completed_updates"] += 1
            clipping["clipped_updates"] += int(was_clipped)
            monitor.advance()

            if sampled_record:
                record = _log_record(
                    step,
                    name,
                    depth,
                    result,
                    aux,
                    loss,
                    grad_norm,
                    schedule,
                    totals,
                    started,
                )
                record.update(diagnostics)
                record["optimizer_clipping"] = {
                    **clipping,
                    "fraction": clipping["clipped_updates"]
                    / clipping["completed_updates"],
                    "threshold": settings.grad_clip,
                    "clip_applied": was_clipped,
                }
                record.update(_cuda_peaks(device))
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                kind = tasks[name].kind if name in tasks else "mixed"
                monitor.record(record, kind, result["targets"], schedule_phase)
                _reset_cuda_peaks(device)
            # A requested stop is handled before any regular checkpoint, so the
            # shutdown writes exactly one full state (latest.pt) inside the
            # provider's grace period. Regular checkpoints due at this step are
            # completed when the run resumes. An explicit boundary also stops
            # before any due held-out assessment; it never shortens the schedule.
            finished = step == settings.total_steps - 1
            boundary_reached = boundary is not None and step + 1 >= boundary
            if boundary_reached or (
                _stop.requested
                and (not finished or validation is not None or test is not None)
            ):
                monitor.phase("saving latest state", force=True)
                path = _save_state(
                    model,
                    optimizer,
                    settings,
                    step,
                    totals,
                    out,
                    rng,
                    tasks,
                    LATEST_STATE,
                )
                reason = "step_boundary" if boundary_reached else "requested"
                raise TrainingInterrupted(step + 1, path, reason=reason)

            if validation is not None and (
                finished
                or (settings.valid_every and (step + 1) % settings.valid_every == 0)
            ):
                assess(validation, "valid", step)
            if finished and test is not None:
                assess(test, "test", step)
            if _stop.requested and (validation is not None or test is not None):
                path = _save_state(
                    model,
                    optimizer,
                    settings,
                    step,
                    totals,
                    out,
                    rng,
                    tasks,
                    LATEST_STATE,
                )
                raise TrainingInterrupted(step + 1, path)
            saved = _maybe_save(
                model,
                optimizer,
                schedule,
                settings,
                step,
                totals,
                out,
                rng,
                tasks,
                on_save=lambda name: monitor.phase(f"saving {name}", force=True),
            )
            if saved:
                last_timed_save = (
                    time.monotonic()
                )  # a regular full state also resets the timer
            elif (
                settings.save_every_minutes > 0
                and not finished
                and time.monotonic() - last_timed_save
                >= settings.save_every_minutes * 60
            ):
                # The timed save writes only the full training state (no public
                # weights): the smallest save that resumes exactly.
                monitor.phase("saving latest state", force=True)
                _save_state(
                    model,
                    optimizer,
                    settings,
                    step,
                    totals,
                    out,
                    rng,
                    tasks,
                    LATEST_STATE,
                )
                last_timed_save = time.monotonic()
    return records


def _merge_micro_batches(
    micro_batches: list, tasks: dict[str, Task], collect_objectives: bool = False
) -> tuple:
    """Summarize the micro-batches of one optimizer step for logging.

    Losses and loss parts are means over the micro-batches that have them
    (the step optimizes the mean loss); tokens, targets and expert counts are
    sums. Task and depth are shown as one value when all micro-batches
    agree, otherwise as "a+b" and a list.
    """
    names = [m[0] for m in micro_batches]
    depths = [m[1] for m in micro_batches]
    results = [m[2] for m in micro_batches]
    auxes = [m[3] for m in micro_batches]
    loss = torch.stack([m[4] for m in micro_batches]).mean()

    objective = torch.stack([r["loss"].detach() for r in results]).mean()
    merged = {
        "loss": objective,
        "tokens": sum(r["tokens"] for r in results),
        "targets": sum(r["targets"] for r in results),
        "batch_shapes": [r["batch_shape"] for r in results],
    }
    task_objectives = {}
    for name in sorted(set(names)) if collect_objectives else ():
        selected = [micro[2] for micro in micro_batches if micro[0] == name]
        components = {}
        for key in ("loss", "denoise", "length", "fast", "reasoning", "cross_mode"):
            values = [
                result[key].detach().float() for result in selected if key in result
            ]
            if values:
                components[key] = torch.stack(values).mean()
        task_objectives[name] = {
            "kind": tasks[name].kind,
            "metrics": components,
            "microbatches": len(selected),
            "rows": sum(result["batch_shape"]["rows"] for result in selected),
            "targets": sum(result["targets"] for result in selected),
        }
    merged["task_objectives"] = task_objectives
    for key in ("denoise", "length", "fast", "reasoning", "cross_mode"):
        values = [r[key] for r in results if key in r]
        if values:
            merged[key] = torch.stack(
                [torch.as_tensor(v, dtype=torch.float32) for v in values]
            ).mean()

    with_aux = [a for a in auxes if a["loss"] is not None]
    if with_aux:
        aux = {
            "loss": torch.stack([a["loss"].detach() for a in with_aux]).mean(),
            "balance": torch.stack([a["balance"] for a in with_aux]).mean(),
            "z": torch.stack([a["z"] for a in with_aux]).mean(),
            "counts": [sum(layer) for layer in zip(*(a["counts"] for a in with_aux))],
        }
    else:
        aux = {"loss": None, "counts": []}

    name = names[0] if len(set(names)) == 1 else "+".join(sorted(set(names)))
    depth = depths[0] if len(set(depths)) == 1 else depths
    return name, depth, merged, aux, loss


def _compile_blocks(model: NuriModel) -> None:
    """torch.compile each block in place, so parameter names and checkpoints do not change.

    dynamic=True avoids a recompile for every new batch length.
    """
    for block in [*model.encoder.blocks, model.core, *model.readout.blocks]:
        block.compile(dynamic=True)


def _task_options(tasks: dict[str, Task]) -> dict:
    return json.loads(json.dumps({name: task.options for name, task in tasks.items()}))


def _random_state(rng: random.Random, tasks: dict[str, Task]) -> dict:
    """Every random source that decides which batches come next."""
    state = {"trainer_rng": rng.getstate(), "tasks": {}}
    for name, task in tasks.items():
        state["tasks"][name] = {
            "rng": task.rng.getstate() if task.rng is not None else None,
            "generator": task.generator.get_state()
            if task.generator is not None
            else None,
        }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


# Settings that may change between the interrupted and the resumed run; they
# do not change what is computed. Everything else must match exactly.
RESUME_FREE_SETTINGS = (
    "out_dir",
    "log_every",
    "save_every",
    "tokenizer_dir",
    "stage",
    "progress",
    "tensorboard_dir",
    "gradient_checkpointing",
    "compile",
    "save_every_minutes",
    "valid_every",
    "stop_after_completed_steps",
)


# Exit code of scripts/train.py after a requested stop (EX_TEMPFAIL: "try again").
INTERRUPTED_EXIT_CODE = 75
LATEST_STATE = "latest"


# Exit code of scripts/train.py when a loss or gradient is NaN/inf (no update, no save).
NONFINITE_EXIT_CODE = 3


class NonFiniteTraining(Exception):
    """A loss or the gradient norm was NaN/inf: the update and any save were skipped."""

    def __init__(self, step: int, task: str):
        super().__init__(
            f"non-finite loss or gradient at step {step} (task {task}); "
            "no update was applied and no state was saved"
        )
        self.step, self.task = step, task


class TrainingInterrupted(Exception):
    """Training stopped early on request; the full state is in `path`."""

    def __init__(self, step: int, path: Path, reason: str = "requested"):
        super().__init__(f"training stopped after step {step}; resume from {path}")
        self.step, self.path, self.reason = step, path, reason


class _StopFlag:
    """A plain boolean, set from a signal handler and read by the training loop.

    Deliberately not a threading.Event: Event.set takes a lock, and a second
    signal arriving while the first handler holds it would deadlock. Setting
    an attribute takes no lock, so the handler is safe to re-enter.
    """

    requested = False


_stop = _StopFlag()


def request_stop() -> None:
    """Ask a running train() to save and stop after the current optimizer step."""
    _stop.requested = True


def _on_sigterm(signum, frame) -> None:
    request_stop()


class _StopOnSigterm:
    """While active, SIGTERM (a preempted cloud VM, `kill`) requests a clean stop.

    Only the main thread may install signal handlers; elsewhere (and where
    SIGTERM does not exist) the handler is simply not installed and
    request_stop() still works.
    """

    def __enter__(self):
        _stop.requested = False
        self.previous = None
        if (
            hasattr(signal, "SIGTERM")
            and threading.current_thread() is threading.main_thread()
        ):
            self.previous = signal.signal(signal.SIGTERM, _on_sigterm)
        return self

    def __exit__(self, *exc):
        if self.previous is not None:
            signal.signal(signal.SIGTERM, self.previous)
        _stop.requested = False
        return False


# Settings added after the first training states were written, with the value
# that matches what older code did.
KERNEL_SETTING_DEFAULTS = {
    "attention_backend": "sdpa",
    "liger_kernels": False,
    "vocab_loss_backend": "torch",
    "vocab_chunk_size": 256,
}
LEGACY_SETTING_DEFAULTS = {
    "grad_accum": 1,
    "native_gqa": False,
    "cross_mode_weight": 0.0,
    **KERNEL_SETTING_DEFAULTS,
}


def _comparable_settings(settings: TrainSettings) -> dict:
    data = {k: v for k, v in asdict(settings).items() if k not in RESUME_FREE_SETTINGS}
    return json.loads(json.dumps(data))  # tuples and lists compare equal after this


def _resume(
    path,
    model,
    optimizer,
    rng: random.Random,
    tasks: dict[str, Task],
    settings: TrainSettings,
) -> tuple[int, dict]:
    state = load_training_state(
        path, model, optimizer
    )  # also checks the config and restores torch/Python RNG
    extra = state["extra"]
    # States written before a setting existed are read with its old behavior.
    saved = {**LEGACY_SETTING_DEFAULTS, **extra["settings"]}
    current = _comparable_settings(settings)
    if saved != current:
        changed = sorted(
            k for k in set(saved) | set(current) if saved.get(k) != current.get(k)
        )
        raise ValueError(
            f"resume needs the same training settings; these differ: {changed}"
        )
    if (
        "task_options" in extra
    ):  # states from before task options existed skip this check
        saved_options, current_options = extra["task_options"], _task_options(tasks)
        if saved_options != current_options:
            changed = sorted(
                n
                for n in set(saved_options) | set(current_options)
                if saved_options.get(n) != current_options.get(n)
            )
            raise ValueError(
                f"resume needs the same batch construction; these tasks differ: {changed}"
            )
    rng.setstate(extra["random"]["trainer_rng"])
    for name, saved in extra["random"]["tasks"].items():
        if saved["rng"] is not None:
            tasks[name].rng.setstate(saved["rng"])
        if saved["generator"] is not None:
            tasks[name].generator.set_state(saved["generator"])
    if "cuda_rng" in extra["random"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(extra["random"]["cuda_rng"])
    return state["step"], dict(extra["totals"])


def _task_loss(model, kind, batch, depth, settings) -> dict:
    vocab_settings = {
        "vocab_loss_backend": settings.vocab_loss_backend,
        "vocab_chunk_size": settings.vocab_chunk_size,
    }
    if kind == "mlm":
        return mlm_loss(model, batch, **vocab_settings)
    if kind == "span":
        return canvas_loss(model, batch, depth, **vocab_settings)
    if kind == "instruction":
        return canvas_loss(
            model, batch, depth, settings.length_weight, **vocab_settings
        )
    if kind == "pairs":
        return pair_loss(
            model,
            batch,
            depth,
            settings.embed_dims,
            settings.temperature,
            settings.cross_mode_weight,
        )
    raise ValueError(f"unknown task kind {kind!r}")


def _log_record(
    step, name, depth, result, aux, loss, grad_norm, schedule, totals, started
) -> dict:
    record = {
        "step": step,
        "task": name,
        "depth": depth,
        "loss": float(loss.detach()),
        "lr": schedule.lr(step),
        "grad_norm": float(grad_norm),
        "tokens": totals["tokens"],
        "targets": totals["targets"],
        "seconds": round(time.time() - started, 1),
    }
    record["objective_loss"] = float(result["loss"].detach())
    record["task_objectives"] = {
        name: {
            **observed,
            "metrics": {
                key: float(value) for key, value in observed["metrics"].items()
            },
        }
        for name, observed in result["task_objectives"].items()
    }
    record["batch_shapes"] = result["batch_shapes"]
    record["batch_summary"] = _summarize_batch_shapes(result["batch_shapes"])
    for key in ("denoise", "length", "fast", "reasoning", "cross_mode"):
        if key in result:
            record[key] = float(result[key])
    if aux["loss"] is not None:
        record["moe_balance"] = float(aux["balance"])
        record["moe_z"] = float(aux["z"])
        record["expert_counts"] = [c.tolist() for c in aux["counts"]]
    return record


def _batch_shape(kind: str, batch: dict, depth: int) -> dict:
    """Numeric batch geometry only; inspect host masks before the GPU transfer."""
    if kind == "pairs":
        views = [
            (batch["query_ids"], batch["query_valid"]),
            (batch["doc_ids"], batch["doc_valid"]),
        ]
    else:
        views = [(batch["input_ids"], batch["input_valid"])]
    rows = views[0][0].shape[0]
    shape = {
        "kind": kind,
        "rows": rows,
        "depth": 0 if kind == "mlm" else depth,
        "input_width": max(ids.shape[1] for ids, _ in views),
        "padded_input_tokens": sum(ids.numel() for ids, _ in views),
        "valid_input_tokens": sum(int(valid.sum()) for _, valid in views),
        "canvas_width": 0,
        "padded_canvas_tokens": 0,
        "valid_canvas_tokens": 0,
    }
    if kind == "pairs":
        shape.update(
            query_width=views[0][0].shape[1],
            document_width=views[1][0].shape[1],
            targets=rows,
        )
    elif kind == "mlm":
        shape["targets"] = int(batch["target_mask"].sum())
    else:
        shape.update(
            canvas_width=batch["canvas"].shape[1],
            padded_canvas_tokens=batch["canvas"].numel(),
            valid_canvas_tokens=int(batch["canvas_valid"].sum()),
            targets=int(batch["loss_mask"].sum()),
        )
    return shape


def _summarize_batch_shapes(shapes: list[dict]) -> dict:
    """One optimizer update: row presentations and tokens sum across microbatches."""
    summary = {
        key: sum(shape[key] for shape in shapes)
        for key in (
            "rows",
            "padded_input_tokens",
            "valid_input_tokens",
            "padded_canvas_tokens",
            "valid_canvas_tokens",
            "targets",
        )
    }
    summary.update(
        microbatches=len(shapes),
        max_rows=max(shape["rows"] for shape in shapes),
        max_input_width=max(shape["input_width"] for shape in shapes),
        max_canvas_width=max(shape["canvas_width"] for shape in shapes),
    )
    padded = summary["padded_input_tokens"] + summary["padded_canvas_tokens"]
    valid = summary["valid_input_tokens"] + summary["valid_canvas_tokens"]
    summary["valid_fraction"] = valid / max(padded, 1)
    return summary


def _reset_cuda_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peaks(device: torch.device) -> dict:
    """Allocator peaks since the previous sampled record, not estimated batch memory."""
    if device.type != "cuda":
        return {}
    return {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _checkpoint_names(schedule, settings, step: int) -> list[str]:
    """The regular checkpoints due after optimizer step `step`."""
    last = step == settings.total_steps - 1
    names = []
    if settings.save_every and (step + 1) % settings.save_every == 0 and not last:
        names.append(f"step-{step + 1}")
    if schedule.is_pre_decay(step):
        names.append("pre-decay")
    if last:
        names.append("final")
    return names


def _complete_checkpoints(
    model, optimizer, schedule, settings, step, totals, out: Path, rng, tasks
) -> None:
    """Write whatever part of the checkpoints due after `step` is missing."""
    for name in _checkpoint_names(schedule, settings, step):
        if not (out / "training_state" / f"{name}.pt").exists():
            _save_state(model, optimizer, settings, step, totals, out, rng, tasks, name)
        if not (out / name / "model.safetensors").exists():
            save_pretrained(model, out / name, tokenizer_dir=settings.tokenizer_dir)


def _maybe_save(
    model,
    optimizer,
    schedule,
    settings,
    step,
    totals,
    out: Path,
    rng,
    tasks,
    on_save: Callable[[str], None] | None = None,
) -> bool:
    """Write the regular checkpoints due after `step`; True if any was written."""
    names = _checkpoint_names(schedule, settings, step)
    for name in names:
        if on_save is not None:
            on_save(name)
        # The full state first, the public weights last: final/model.safetensors
        # is the runner's "stage finished" marker, so it may only appear once
        # final.pt is safely on disk.
        _save_state(model, optimizer, settings, step, totals, out, rng, tasks, name)
        save_pretrained(model, out / name, tokenizer_dir=settings.tokenizer_dir)
    return bool(names)


def _save_state(
    model, optimizer, settings, step, totals, out: Path, rng, tasks, name: str
) -> Path:
    """Write training_state/<name>.pt for resuming after `step` (atomic)."""
    extra = {
        "totals": dict(totals),
        "random": _random_state(rng, tasks),
        "settings": _comparable_settings(settings),
        "task_options": _task_options(tasks),
    }
    path = out / "training_state" / f"{name}.pt"
    save_training_state(path, model, optimizer, step + 1, extra)
    return path
