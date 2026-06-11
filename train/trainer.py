"""Training loop for SamatNext-520M per SPEC.md."""

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from data.dataset import BatchOutput, PythonCodeDataset
from model.config import DEFAULT_CONFIG, SamatNextConfig
from model.model import SamatNext
from train.optimizer import build_optimizer
from train.scheduler import CosineWarmupScheduler


BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
MAX_STEPS = 100000
SAVE_EVERY = 1000
EVAL_EVERY = 500
GRAD_CLIP = 1.0
LOG_EVERY = 10


class Trainer:
    def __init__(
        self,
        model: SamatNext,
        config: SamatNextConfig,
        output_dir: str = "checkpoints",
    ):
        self.model = model
        self.config = config
        self.optimizer = build_optimizer(model, config)
        self.scheduler = CosineWarmupScheduler(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.best_loss = float("inf")

    def loss_fn(
        self,
        model,
        input_ids: mx.array,
        targets: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        out = model(input_ids, targets=targets, loss_mask=loss_mask)
        return out["loss"]

    def train_step(self, batches: list[BatchOutput]) -> dict:
        if not batches:
            raise ValueError("train_step requires at least one microbatch")

        accumulated_grads = None
        accumulated_loss = 0.0
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

        for batch in batches:
            input_ids = mx.array(batch.input_ids, dtype=mx.int32)
            targets = mx.array(batch.targets, dtype=mx.int32)
            loss_mask = mx.array(batch.loss_mask, dtype=mx.bfloat16)
            loss, grads = loss_and_grad_fn(
                self.model,
                input_ids,
                targets,
                loss_mask,
            )
            mx.eval(loss, grads)
            accumulated_loss += loss.item()
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda total, grad: total + grad,
                    accumulated_grads,
                    grads,
                )
                mx.eval(accumulated_grads)

        scale = 1.0 / len(batches)
        averaged_grads = tree_map(lambda grad: grad * scale, accumulated_grads)
        averaged_grads, total_norm = optim.clip_grad_norm(
            averaged_grads,
            max_norm=GRAD_CLIP,
        )

        lr = self.scheduler.get_lr(self.step)
        self.optimizer.learning_rate = lr
        self.optimizer.update(self.model, averaged_grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        return {
            "loss": accumulated_loss * scale,
            "lr": lr,
            "grad_norm": total_norm.item(),
            "microbatches": len(batches),
        }

    def save_checkpoint(self, step: int, loss: float) -> None:
        path = self.output_dir / f"step_{step:06d}.npz"
        flat_params = dict(tree_flatten(self.model.parameters()))
        mx.savez(str(path), **flat_params)
        print(f"Saved checkpoint: {path} (loss={loss:.4f})")

    def load_checkpoint(self, path: str) -> None:
        weights = mx.load(path)
        self.model.load_weights(list(weights.items()))
        print(f"Loaded checkpoint: {path}")

    def train(
        self,
        dataset: PythonCodeDataset,
        resume_from: str = None,
    ) -> None:
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"Starting training from step {self.step}")
        print(f"Output dir: {self.output_dir}")
        grad_accum_loss = 0.0
        grad_accum_count = 0
        t0 = time.time()

        pending_batches = []
        while self.step < MAX_STEPS:
            batches_seen = 0
            for batch in dataset.iterate_batches(BATCH_SIZE):
                batches_seen += 1
                if self.step >= MAX_STEPS:
                    break

                pending_batches.append(batch)
                if len(pending_batches) < GRAD_ACCUM_STEPS:
                    continue

                metrics = self.train_step(pending_batches)
                pending_batches = []
                grad_accum_loss += metrics["loss"]
                grad_accum_count += 1

                if self.step % LOG_EVERY == 0:
                    elapsed = time.time() - t0
                    avg_loss = grad_accum_loss / max(1, grad_accum_count)
                    print(
                        f"step={self.step:06d} loss={avg_loss:.4f} "
                        f"lr={metrics['lr']:.2e} "
                        f"grad_norm={metrics['grad_norm']:.3f} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    grad_accum_loss = 0.0
                    grad_accum_count = 0

                if self.step % SAVE_EVERY == 0 and self.step > 0:
                    self.save_checkpoint(self.step, metrics["loss"])

                self.step += 1

            if batches_seen == 0:
                raise RuntimeError(
                    "Dataset produced no complete microbatches; "
                    "add more samples or reduce BATCH_SIZE"
                )

        print("Training complete")


def check_trainer(config: SamatNextConfig) -> None:
    print("Trainer: skipping full check (requires trained tokenizer + dataset)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Grad accum steps: {GRAD_ACCUM_STEPS}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Save every: {SAVE_EVERY}")
    print("Trainer OK")


if __name__ == "__main__":
    check_trainer(DEFAULT_CONFIG)
