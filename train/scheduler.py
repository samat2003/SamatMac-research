"""Cosine LR scheduler with warmup for SamatNext-520M per SPEC.md."""

import math

from model.config import DEFAULT_CONFIG, SamatNextConfig


class CosineWarmupScheduler:
    def __init__(self, config: SamatNextConfig):
        self.base_lr = 3e-4
        self.warmup_steps = config.warmup_steps
        self.max_steps = 100000
        self.min_lr = self.base_lr * 0.1

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


def check_scheduler(config: SamatNextConfig) -> None:
    sched = CosineWarmupScheduler(config)
    assert sched.get_lr(0) == 0.0
    assert abs(sched.get_lr(config.warmup_steps) - 3e-4) < 1e-10
    assert sched.get_lr(100000) == sched.min_lr
    print("Scheduler OK")


if __name__ == "__main__":
    check_scheduler(DEFAULT_CONFIG)
