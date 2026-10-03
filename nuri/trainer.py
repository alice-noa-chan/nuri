"""Training loop and the loss for each kind of batch (ARCHITECTURE.md §8-C, §11).

A training run mixes several batch "tasks" (for example MLM and span
denoising in the base stage). Each step picks one task by its mix weight,
builds a batch, computes that task's loss and takes one optimizer step.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .checkpoint import load_training_state, save_pretrained, save_training_state
from .losses import contrastive_loss, length_loss, sample_mean_ce, token_mean_ce
from .model import NuriModel
from .monitoring import TrainingMonitor
from .schedule import WSDSchedule

# ----- loss for each batch kind ------------------------------------------------------


def mlm_loss(model: NuriModel, batch: dict) -> dict:
    """Input MLM: predicts masked input tokens straight from X (no core, no readout)."""
    x, _ = model.encode(batch["input_ids"], batch["input_valid"])
    mask = batch["target_mask"]
    loss = token_mean_ce(model.mlm_logits(x, mask), batch["labels"][mask])
    return {"loss": loss, "tokens": int(batch["input_valid"].sum()), "targets": int(mask.sum())}


def canvas_loss(model: NuriModel, batch: dict, depth: int, length_weight: float = 0.0) -> dict:
    """Span denoising and instruction tuning: fill the masked canvas from [X; R].

    The length loss is added only for batches with natural answer lengths
    (instruction tuning). Random span lengths must never train it (§7).
    """
    x, valid = model.encode(batch["input_ids"], batch["input_valid"])
    r = model.reason(x, valid, depth)
    cache = model.generation_memory(x, valid, r)
    hidden = model.readout_hidden(batch["canvas"], batch["canvas_valid"], cache)
    mask = batch["loss_mask"]
    loss = sample_mean_ce(model.vocab_logits(hidden[mask]), batch["targets"], mask)
    result = {"denoise": loss.detach()}
    if length_weight:
        u, r_mean = model.pool(x, valid, r)
        length = length_loss(model.length_logits(u, r_mean), batch["lengths"])
        loss = loss + length_weight * length
        result["length"] = length.detach()
    tokens = int(batch["input_valid"].sum() + batch["canvas_valid"].sum())
    return dict(result, loss=loss, tokens=tokens, targets=int(mask.sum()))


def pair_loss(model: NuriModel, batch: dict, depth: int, dims: tuple[int, ...], temperature: float) -> dict:
    """Contrastive embedding loss, trained for both the fast and the reasoning mode.

    Query and document always use the same mode, as the spec's default
    retrieval does (§6). Each side is encoded once and its X is reused for
    both modes, so the encoder runs twice per step, not four times.
    """
    xq, vq = model.encode(batch["query_ids"], batch["query_valid"])
    xd, vd = model.encode(batch["doc_ids"], batch["doc_valid"])
    losses = {}
    for mode in ("fast", "reasoning"):
        q = model.embed_memory(xq, vq, mode=mode, steps=depth)
        d = model.embed_memory(xd, vd, mode=mode, steps=depth)
        losses[mode] = contrastive_loss(q, d, temperature, dims)
    tokens = int(batch["query_valid"].sum() + batch["doc_valid"].sum())
    return {"loss": (losses["fast"] + losses["reasoning"]) / 2, "fast": losses["fast"].detach(),
            "reasoning": losses["reasoning"].detach(), "tokens": tokens, "targets": batch["query_ids"].shape[0]}


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
    save_every: int = 0            # 0 = only pre-decay and final checkpoints
    seed: int = 0
    length_weight: float = 0.1     # used by "instruction" batches
    embed_dims: tuple[int, ...] = (64, 128, 256)
    temperature: float = 0.05
    mix: dict[str, float] = field(default_factory=dict)
    tokenizer_dir: str | None = None  # copied into every saved checkpoint
    stage: str = "training"           # label only; the task mix defines the computation
    progress: bool = True
    tensorboard_dir: str | None = None  # None = no event files or TensorBoard import
    grad_accum: int = 1               # micro-batches per optimizer step (each draws its own task and depth)
    gradient_checkpointing: bool = False  # recompute block activations in backward (same results, less memory)
    compile: bool = False             # torch.compile the blocks in place (same weights and names)


@dataclass
class Task:
    """A named batch source. kind is "mlm", "span", "instruction" or "pairs".

    rng and generator are the random sources next_batch draws from. The
    trainer saves and restores them, so a resumed run sees the same batches
    as an uninterrupted one.
    """
    kind: str
    next_batch: Callable[[], dict]
    rng: random.Random | None = None
    generator: torch.Generator | None = None


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model: NuriModel, settings: TrainSettings) -> torch.optim.AdamW:
    """AdamW without weight decay on norms, gates, slots and the embedding."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2 or "embedding" in name or "slots" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [{"params": decay, "weight_decay": settings.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=settings.peak_lr, betas=settings.betas)


def train(model: NuriModel, tasks: dict[str, Task], settings: TrainSettings, device=None,
          resume_from: str | Path | None = None) -> list[dict]:
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
    device = device or default_device()
    model.to(device).train()
    model.set_gradient_checkpointing(settings.gradient_checkpointing)
    if settings.compile:
        _compile_blocks(model)
    out = Path(settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_settings.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")

    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)
    optimizer = make_optimizer(model, settings)
    schedule = WSDSchedule(settings.peak_lr, settings.total_steps, settings.warmup_steps, settings.decay_fraction)
    names = list(settings.mix)
    weights = [settings.mix[n] for n in names]
    use_bf16 = device.type == "cuda"
    totals = {"tokens": 0, "targets": 0}
    start_step = 0
    if resume_from is not None:
        start_step, totals = _resume(resume_from, model, optimizer, rng, tasks, settings)
    records, started = [], time.time()

    with TrainingMonitor(settings.total_steps, start_step, totals["tokens"],
                         stage=settings.stage, progress=settings.progress,
                         tensorboard_dir=settings.tensorboard_dir,
                         resuming=resume_from is not None) as monitor, \
            open(out / "train_log.jsonl", "a", encoding="utf-8") as log:
        for step in range(start_step, settings.total_steps):
            for group in optimizer.param_groups:
                group["lr"] = schedule.lr(step)
            schedule_phase = "warmup" if step < schedule.warmup_steps else (
                "stable" if step < schedule.decay_start else "decay")
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            # With grad_accum > 1 the gradients of several micro-batches add up
            # before one optimizer step; each loss is divided so the step uses
            # their mean. grad_accum = 1 is exactly the plain loop.
            for micro in range(settings.grad_accum):
                name = rng.choices(names, weights)[0]
                depth = rng.choice(model.config.train_depths)
                if micro == 0:
                    monitor.begin_step(name, depth if tasks[name].kind != "mlm" else 0, schedule_phase)
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in tasks[name].next_batch().items()}

                monitor.phase("forward")
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    result = _task_loss(model, tasks[name].kind, batch, depth, settings)
                    aux = model.pop_moe_aux()
                    loss = result["loss"] + (aux["loss"] if aux["loss"] is not None else 0.0)

                monitor.phase("backward")
                (loss / settings.grad_accum).backward()
                step_losses.append(loss.detach())
                totals["tokens"] += result["tokens"]
                totals["targets"] += result["targets"]
            loss = torch.stack(step_losses).mean()  # logged value: mean over the micro-batches

            monitor.phase("optimizer")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            monitor.advance()

            if step % settings.log_every == 0 or step == settings.total_steps - 1:
                record = _log_record(step, name, depth, result, aux, loss, grad_norm, schedule, totals, started)
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                monitor.record(record, tasks[name].kind, result["targets"], schedule_phase)
            _maybe_save(model, optimizer, schedule, settings, step, totals, out, rng, tasks,
                        on_save=lambda name: monitor.phase(f"saving {name}", force=True))
    return records


def _compile_blocks(model: NuriModel) -> None:
    """torch.compile each block in place, so parameter names and checkpoints do not change.

    dynamic=True avoids a recompile for every new batch length.
    """
    for block in [*model.encoder.blocks, model.core, *model.readout.blocks]:
        block.compile(dynamic=True)


def _random_state(rng: random.Random, tasks: dict[str, Task]) -> dict:
    """Every random source that decides which batches come next."""
    state = {"trainer_rng": rng.getstate(), "tasks": {}}
    for name, task in tasks.items():
        state["tasks"][name] = {
            "rng": task.rng.getstate() if task.rng is not None else None,
            "generator": task.generator.get_state() if task.generator is not None else None,
        }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


# Settings that may change between the interrupted and the resumed run; they
# do not change what is computed. Everything else must match exactly.
RESUME_FREE_SETTINGS = ("out_dir", "log_every", "save_every", "tokenizer_dir",
                        "stage", "progress", "tensorboard_dir", "gradient_checkpointing", "compile")


def _comparable_settings(settings: TrainSettings) -> dict:
    data = {k: v for k, v in asdict(settings).items() if k not in RESUME_FREE_SETTINGS}
    return json.loads(json.dumps(data))  # tuples and lists compare equal after this


def _resume(path, model, optimizer, rng: random.Random, tasks: dict[str, Task],
            settings: TrainSettings) -> tuple[int, dict]:
    state = load_training_state(path, model, optimizer)  # also checks the config and restores torch/Python RNG
    extra = state["extra"]
    saved, current = extra["settings"], _comparable_settings(settings)
    if saved != current:
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        raise ValueError(f"resume needs the same training settings; these differ: {changed}")
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
    if kind == "mlm":
        return mlm_loss(model, batch)
    if kind == "span":
        return canvas_loss(model, batch, depth)
    if kind == "instruction":
        return canvas_loss(model, batch, depth, settings.length_weight)
    if kind == "pairs":
        return pair_loss(model, batch, depth, settings.embed_dims, settings.temperature)
    raise ValueError(f"unknown task kind {kind!r}")


def _log_record(step, name, depth, result, aux, loss, grad_norm, schedule, totals, started) -> dict:
    record = {"step": step, "task": name, "depth": depth, "loss": float(loss.detach()), "lr": schedule.lr(step),
              "grad_norm": float(grad_norm), "tokens": totals["tokens"], "targets": totals["targets"],
              "seconds": round(time.time() - started, 1)}
    for key in ("denoise", "length", "fast", "reasoning"):
        if key in result:
            record[key] = float(result[key])
    if aux["loss"] is not None:
        record["moe_balance"] = float(aux["balance"])
        record["moe_z"] = float(aux["z"])
        record["expert_counts"] = [c.tolist() for c in aux["counts"]]
    return record


def _maybe_save(model, optimizer, schedule, settings, step, totals, out: Path, rng, tasks,
                on_save: Callable[[str], None] | None = None) -> None:
    last = step == settings.total_steps - 1
    names = []
    if settings.save_every and (step + 1) % settings.save_every == 0 and not last:
        names.append(f"step-{step + 1}")
    if schedule.is_pre_decay(step):
        names.append("pre-decay")
    if last:
        names.append("final")
    for name in names:
        if on_save is not None:
            on_save(name)
        save_pretrained(model, out / name, tokenizer_dir=settings.tokenizer_dir)
        extra = {"totals": dict(totals), "random": _random_state(rng, tasks),
                 "settings": _comparable_settings(settings)}
        save_training_state(out / "training_state" / f"{name}.pt", model, optimizer, step + 1, extra)
