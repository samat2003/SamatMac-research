"""AdamW optimizer for SamatNext-520M per SPEC.md."""

import mlx.core as mx
import mlx.optimizers as optim

from model.config import DEFAULT_CONFIG, SamatNextConfig


def build_optimizer(model, config: SamatNextConfig) -> optim.AdamW:
    return optim.AdamW(
        learning_rate=3e-4,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=0.1,
    )
