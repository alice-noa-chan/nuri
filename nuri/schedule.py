"""Warmup-Stable-Decay learning rate schedule (ARCHITECTURE.md §15).

  warmup: 0 -> peak, linear
  stable: peak
  decay:  peak -> peak * min_ratio, linear, over the last `decay_fraction` of steps

The checkpoint saved right before the decay starts ("pre-decay") is a good
starting point for continued pretraining, because the LR was never lowered.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WSDSchedule:
    peak_lr: float
    total_steps: int
    warmup_steps: int
    decay_fraction: float = 0.15
    min_ratio: float = 0.0

    @property
    def decay_start(self) -> int:
        return self.total_steps - int(round(self.total_steps * self.decay_fraction))

    def lr(self, step: int) -> float:
        """Learning rate for a 0-based step."""
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step < self.decay_start:
            return self.peak_lr
        progress = (step - self.decay_start + 1) / max(1, self.total_steps - self.decay_start)
        return self.peak_lr * (1 - (1 - self.min_ratio) * min(1.0, progress))

    def is_pre_decay(self, step: int) -> bool:
        """True for the last stable step, where the pre-decay checkpoint is saved."""
        return step == self.decay_start - 1
