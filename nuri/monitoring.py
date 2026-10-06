"""Terminal progress and optional TensorBoard metrics for one training run."""

from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm


def _wait_for_resume_event_order(directory: str | Path):
    """Keep a resumed event file after older files in TensorBoard's name order."""
    timestamps = []
    for path in Path(directory).glob("events.out.tfevents.*"):
        timestamp = path.name.split(".")[3]
        if not path.is_file() or not timestamp.isascii() or not timestamp.isdigit():
            raise RuntimeError("TensorBoard event filenames need a numeric timestamp; "
                               "choose a new tensorboard_dir.")
        timestamps.append(int(timestamp))
    if not timestamps:
        return
    latest = max(timestamps)
    # TensorBoard filenames use whole seconds, then host/PID/unpadded counter.
    # A later second prevents counter 10 from sorting before counter 9.
    now = time.time()
    if latest > int(now):
        raise RuntimeError("TensorBoard events are dated in the future; correct the system clock "
                           "or choose a new tensorboard_dir.")
    if latest == int(now):
        time.sleep(latest + 1 - now + 0.001)
    if int(time.time()) <= latest:
        raise RuntimeError("The TensorBoard filename clock did not advance; correct the system clock "
                           "or choose a new tensorboard_dir.")


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
            if resuming:
                _wait_for_resume_event_order(tensorboard_dir)
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

    def begin_step(self, task: str, depth: int, schedule_phase: str,
                   micro: int = 1, micro_count: int = 1):
        self.context = f"{self.stage}/{task} T={depth} {schedule_phase}"
        if micro_count > 1:
            self.context += f" micro={micro}/{micro_count}"
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
        for key in ("denoise", "length", "fast", "reasoning", "cross_mode", "moe_balance", "moe_z"):
            if key in record:
                metrics[key] = f"{record[key]:.4f}"
        clipping = record.get("optimizer_clipping", {})
        if clipping:
            metrics["clip"] = f"{clipping['fraction']:.0%}"
        summary = record.get("batch_summary", {})
        if summary:
            metrics.update(rows=summary["rows"],
                           shape=f"{summary['max_input_width']}/{summary['max_canvas_width']}",
                           valid=f"{summary['valid_fraction']:.0%}")
        if "cuda_peak_allocated_bytes" in record:
            metrics["peak GiB"] = f"{record['cuda_peak_allocated_bytes'] / 2**30:.2f}"
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
            "train/schedule_phase": {"warmup": 0, "stable": 1, "decay": 2}[schedule_phase],
        }
        if clipping:
            for tag, key in {
                "clip_count": "clipped_updates", "clip_observed_updates": "completed_updates",
                "clip_fraction": "fraction", "clip_window_start_step": "start_completed_step",
                "clip_threshold": "threshold", "clip_applied": "clip_applied",
            }.items():
                scalars[f"optimizer/{tag}"] = clipping[key]
        for group in ("encoder", "core", "readout", "tied_embedding", "other"):
            if group in record.get("pre_clip_grad_norms", {}):
                scalars[f"optimizer/pre_clip_grad_norm/{group}"] = record["pre_clip_grad_norms"][group]
        for statistic in ("min", "mean", "median", "max", "rows"):
            if statistic in record.get("canvas_position_row_norms", {}):
                scalars[f"parameters/canvas_position_row_norm/{statistic}"] = record["canvas_position_row_norms"][statistic]
        objectives = record.get("task_objectives") or {
            task: {"kind": kind, "metrics": {"loss": record.get("objective_loss", record["loss"]),
                   **{key: record[key] for key in ("denoise", "length", "fast", "reasoning", "cross_mode")
                      if key in record}}}}
        for name, observed in objectives.items():
            label = name if name in {"mlm", "span", "instruction", "reasoning", "pairs"} else observed["kind"]
            for key, value in observed["metrics"].items():
                tag = "objective_loss" if key == "loss" else key
                scalars[f"train/{label}/{tag}"] = value
            for key in ("microbatches", "rows", "targets"):
                if key in observed:
                    scalars[f"train/{label}/observed_{key}"] = observed[key]
        depth = record["depth"] if kind != "mlm" else 0
        if isinstance(depth, (tuple, list)):
            # An accumulated update can sample several recurrent depths.
            # Keep the range explicit instead of labeling it as one depth.
            scalars["train/reasoning_depth_mean"] = sum(depth) / len(depth)
            scalars["train/reasoning_depth_min"] = min(depth)
            scalars["train/reasoning_depth_max"] = max(depth)
        else:
            scalars["train/reasoning_depth"] = depth
        for key in ("denoise", "length", "fast", "reasoning", "cross_mode"):
            if key in record:
                scalars[f"loss/{task}/{key}"] = record[key]
        for key in ("moe_balance", "moe_z"):
            if key in record:
                scalars[f"moe/{key.removeprefix('moe_')}"] = record[key]
        for key, value in summary.items():
            scalars[f"batch/{key}"] = value
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            if key in record:
                scalars[f"memory/{key}"] = record[key]
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)
        for layer, counts in enumerate(record.get("expert_counts", [])):
            for expert, count in enumerate(counts):
                self.writer.add_scalar(f"moe/layer_{layer}/expert_{expert}_assignments", count, step)

    def record_holdout(self, report: dict):
        """Write objective-only holdout curves and numeric assessed coverage."""
        split, step = report["split"], report["step"]
        metrics = {f"{split}/{name}": f"{task['metrics']['loss']:.4f}" for name, task in report["tasks"].items()}
        self.bar.set_postfix(metrics, refresh=False)
        if self.writer is None:
            return
        for key in ("assessed_rows", "assessed_batches", "seconds", "seed", "depth"):
            self.writer.add_scalar(f"{split}/{key}", report[key], step)
        # Only fixed diagnostic labels and digests are text, never input text,
        # source names, arguments or file paths.
        self.writer.add_text(f"{split}/coverage", report["coverage"], step)
        self.writer.add_text(f"{split}/policy_sha256", report["policy_sha256"], step)
        for name, task_report in report["tasks"].items():
            task = name if name in {"mlm", "span", "instruction", "reasoning", "pairs"} else task_report["kind"]
            language, separator, objective = name.partition("/")
            if (separator and language in {"ko", "en", "ja"}
                    and objective in {"mlm", "span", "instruction", "reasoning", "pairs"}):
                task = name
            prefix = f"{split}/{task}"
            for key, value in task_report["metrics"].items():
                tag = "objective_loss" if key == "loss" else key
                self.writer.add_scalar(f"{prefix}/{tag}", value, step)
            for category in ("loss_numerators", "loss_denominators", "counts"):
                for key, value in task_report[category].items():
                    self.writer.add_scalar(f"{prefix}/{category}/{key}", value, step)
            self.writer.add_scalar(f"{prefix}/assessed_batches", task_report["batches"], step)
            for key in ("batch_sha256", "target_sha256"):
                self.writer.add_text(f"{prefix}/{key}", task_report[key], step)
        self.writer.flush()
