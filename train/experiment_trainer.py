"""Shared training loop for the 20M Validation Experiment."""

import time
import math
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from data.dataset import BatchOutput, PythonCodeDataset
from model.config_20m import Experiment20MConfig

# Constants for the experiment
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4   # 4 microbatches × BS=4 = effective batch size 16
MAX_STEPS = 1000      # Reduced to finish full run in a few hours locally
SAVE_EVERY = 1000
EVAL_EVERY = 500
GRAD_CLIP = 1.0
LOG_EVERY = 10


def build_experiment_optimizer() -> optim.AdamW:
    return optim.AdamW(
        learning_rate=3e-4,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=0.1,
    )


class ExperimentScheduler:
    def __init__(self, config: Experiment20MConfig, max_steps: int):
        self.base_lr = 3e-4
        self.warmup_steps = config.warmup_steps
        self.max_steps = max_steps
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


class ExperimentTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Experiment20MConfig,
        output_dir: str,
        seed: int,
        max_steps: int = MAX_STEPS,
    ):
        self.model = model
        self.config = config
        self.optimizer = build_experiment_optimizer()
        self.scheduler = ExperimentScheduler(config, max_steps=max_steps)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.max_steps = max_steps
        self.step = 0
        
        self.log_file = self.output_dir / "train_logs.jsonl"

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

        for i, batch in enumerate(batches):
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

    def save_checkpoint(self, step: int) -> None:
        path = self.output_dir / f"step_{step:06d}.npz"
        flat_params = dict(tree_flatten(self.model.parameters()))
        mx.savez(str(path), **flat_params)

    def log_metrics(self, step: int, metrics: dict) -> None:
        metrics["step"] = step
        metrics["seed"] = self.seed
        metrics["model"] = self.config.model_name
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def train(self, dataset: PythonCodeDataset) -> None:
        print(f"Starting training {self.config.model_name} (seed {self.seed})")
        print(f"Output dir: {self.output_dir}")
        grad_accum_loss = 0.0
        grad_accum_count = 0
        t0 = time.time()

        pending_batches = []
        while self.step < self.max_steps:
            batches_seen = 0
            for batch in dataset.iterate_batches(BATCH_SIZE):
                batches_seen += 1
                if self.step >= self.max_steps:
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
                    
                    self.log_metrics(self.step, {
                        "loss": avg_loss,
                        "lr": metrics['lr'],
                        "grad_norm": metrics['grad_norm']
                    })
                    
                    print(
                        f"step={self.step:06d} loss={avg_loss:.4f} "
                        f"lr={metrics['lr']:.2e} "
                        f"grad_norm={metrics['grad_norm']:.3f} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    grad_accum_loss = 0.0
                    grad_accum_count = 0

                if self.step % SAVE_EVERY == 0 and self.step > 0:
                    self.save_checkpoint(self.step)

                self.step += 1

            if batches_seen == 0:
                raise RuntimeError("Dataset produced no complete microbatches")

        self.save_checkpoint(self.step)
        print(f"Training complete for {self.config.model_name} (seed {self.seed})")
