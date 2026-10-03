"""Terminal progress and optional TensorBoard metrics for one training run."""

from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm


class TrainingMonitor:
    """Observe completed updates without touching tensors or random state.

    Losses arrive as the Python numbers already collected for the JSON log.
    Throughput counts only work done since the previous measurement in this
    process, including time spent building batches and saving checkpoints.
    """

    def __init__(self, total_steps: int, start_step: int, start_tokens: int,
                 stage: str = "training", progress: bool = True,
                 tensorboard_dir: str | Path | None = None, resuming: bool = False):
        self.stage = stage
        self.context = stage
        self.last_step = start_step
        self.last_tokens = start_tokens
        self.first_step = True
        self.writer = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError('TensorBoard needs: python -m pip install -e ".[tensorboard]"') from error
            # The checkpoint holds completed steps. Hide later, stale points
            # when the same event directory is used after an interruption.
            self.writer = SummaryWriter(str(tensorboard_dir),
                                        purge_step=start_step + 1 if resuming else None,
                                        max_queue=1000, flush_secs=10)
        self.bar = tqdm(total=total_steps, initial=start_step, desc=stage,
                        unit="step", dynamic_ncols=True, ascii=True,
                        mininterval=0.5, disable=not progress)
        self.last_measurement = time.perf_counter()
        self.last_refresh = self.last_measurement

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.phase("done" if exc_type is None else "interrupted", force=True)
        finally:
            try:
                if self.writer is not None:
                    self.writer.close()  # flush even when training fails
            finally:
                self.bar.close()

    def begin_step(self, task: str, depth: int, schedule_phase: str):
        self.context = f"{self.stage}/{task} T={depth} {schedule_phase}"
        self.phase("batch")

    def phase(self, operation: str, force: bool = False):
        """Update the current operation, throttling terminal writes to 2 Hz."""
        self.bar.set_description_str(f"{self.context} | {operation}", refresh=False)
        now = time.perf_counter()
        if force or self.first_step or now - self.last_refresh >= 0.5:
            self.bar.refresh()
            self.last_refresh = now

    def advance(self):
        """Called exactly once after a successful optimizer update."""
        self.first_step = False
        self.bar.update(1)

    def record(self, record: dict, kind: str, batch_targets: int, schedule_phase: str):
        """Display and write the sampled metrics, using completed-step indices."""
        step = record["step"] + 1
        now = time.perf_counter()
        elapsed = max(now - self.last_measurement, 1e-9)
        tokens_per_second = (record["tokens"] - self.last_tokens) / elapsed
        steps_per_second = (step - self.last_step) / elapsed
        self.last_measurement, self.last_tokens, self.last_step = now, record["tokens"], step
        metrics = {"loss": f"{record['loss']:.4f}", "lr": f"{record['lr']:.2e}",
                   "grad": f"{record['grad_norm']:.2f}", "tokens": f"{record['tokens']:,}",
                   "tok/s": f"{tokens_per_second:.0f}"}
        for key in ("denoise", "length", "fast", "reasoning", "moe_balance", "moe_z"):
            if key in record:
                metrics[key] = f"{record[key]:.4f}"
        self.bar.set_postfix(metrics, refresh=False)
        if self.writer is None:
            return

        # Explicit numeric tags avoid logging arguments, input text or paths.
        # Known pipeline tasks get separate curves; custom sources use kind.
        task = record["task"] if record["task"] in {"mlm", "span", "instruction", "reasoning", "pairs"} else kind
        scalars = {
            "loss/total": record["loss"], f"loss/{task}/total": record["loss"],
            "optimizer/lr": record["lr"], "optimizer/grad_norm": record["grad_norm"],
            "data/processed_tokens": record["tokens"], f"batch/{kind}/targets": batch_targets,
            "performance/tokens_per_second": tokens_per_second,
            "performance/steps_per_second": steps_per_second,
            "train/reasoning_depth": record["depth"] if kind != "mlm" else 0,
            "train/schedule_phase": {"warmup": 0, "stable": 1, "decay": 2}[schedule_phase],
        }
        for key in ("denoise", "length", "fast", "reasoning"):
            if key in record:
                scalars[f"loss/{task}/{key}"] = record[key]
        for key in ("moe_balance", "moe_z"):
            if key in record:
                scalars[f"moe/{key.removeprefix('moe_')}"] = record[key]
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)
        for layer, counts in enumerate(record.get("expert_counts", [])):
            for expert, count in enumerate(counts):
                self.writer.add_scalar(f"moe/layer_{layer}/expert_{expert}_assignments", count, step)
