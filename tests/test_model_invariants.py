"""Model and optimizer invariants for SamatNext-520M per SPEC.md."""

import tempfile
import unittest

import mlx.core as mx
import numpy as np

from data.dataset import BatchOutput
from model.config import DEFAULT_CONFIG, SamatNextConfig
from model.memory_bus import MemoryBus
from model.model import SamatNext, count_params_exact
from train.trainer import Trainer


def tiny_config() -> SamatNextConfig:
    return SamatNextConfig(
        d_model=16,
        n_layers=4,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=32,
        vocab_size=64,
        max_seq_len=16,
        memory_bus_layers=1,
        n_latent_tokens=1,
        mtp_heads=2,
    )


class ModelInvariantTest(unittest.TestCase):
    def test_reported_parameter_count_matches_checkpoint(self):
        model_stub = type("ModelStub", (), {"config": DEFAULT_CONFIG})()
        self.assertEqual(count_params_exact(model_stub), 520_930_316)

    def test_memory_bus_is_causal(self):
        config = tiny_config()
        bus = MemoryBus(config)
        prefix = mx.arange(3 * config.d_model).reshape(1, 3, config.d_model)
        future_a = mx.zeros((1, 2, config.d_model))
        future_b = mx.ones((1, 2, config.d_model)) * 100
        x_a = mx.concatenate([prefix, future_a], axis=1).astype(mx.bfloat16)
        x_b = mx.concatenate([prefix, future_b], axis=1).astype(mx.bfloat16)

        y_a = bus(x_a, do_write=True)
        mx.eval(y_a)
        bus.reset()
        y_b = bus(x_b, do_write=True)
        mx.eval(y_b)

        self.assertTrue(bool(mx.allclose(y_a[:, :3], y_b[:, :3]).item()))

    def test_complete_model_is_causal(self):
        config = tiny_config()
        model = SamatNext(config)
        input_a = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
        input_b = mx.array([[1, 2, 3, 40, 41, 42]], dtype=mx.int32)
        logits_a = model(input_a)["logits"]
        logits_b = model(input_b)["logits"]
        mx.eval(logits_a, logits_b)

        self.assertTrue(bool(mx.allclose(logits_a[:, :3], logits_b[:, :3]).item()))

    def test_zero_mask_produces_zero_training_loss(self):
        config = tiny_config()
        model = SamatNext(config)
        input_ids = mx.zeros((1, 8), dtype=mx.int32)
        targets = mx.ones((1, 8), dtype=mx.int32)
        loss_mask = mx.zeros((1, 8), dtype=mx.bfloat16)
        out = model(input_ids, targets=targets, loss_mask=loss_mask)
        mx.eval(out["loss"])

        self.assertAlmostEqual(out["loss"].item(), 0.0, places=6)

    def test_masked_targets_do_not_change_loss(self):
        config = tiny_config()
        model = SamatNext(config)
        input_ids = mx.zeros((1, 8), dtype=mx.int32)
        targets_a = mx.ones((1, 8), dtype=mx.int32)
        targets_b = mx.array([[1, 1, 1, 1, 7, 8, 9, 10]], dtype=mx.int32)
        loss_mask = mx.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=mx.bfloat16)
        loss_a = model(input_ids, targets=targets_a, loss_mask=loss_mask)["loss"]
        loss_b = model(input_ids, targets=targets_b, loss_mask=loss_mask)["loss"]
        mx.eval(loss_a, loss_b)

        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=5)

    def test_accumulated_microbatches_apply_one_optimizer_update(self):
        config = tiny_config()
        model = SamatNext(config)
        batch = BatchOutput(
            input_ids=np.zeros((1, 8), dtype=np.int32),
            targets=np.ones((1, 8), dtype=np.int32),
            loss_mask=np.ones((1, 8), dtype=np.float32),
            is_fim=[False],
        )
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = Trainer(model, config, output_dir=output_dir)
            metrics = trainer.train_step([batch, batch])
            mx.eval(trainer.optimizer.state)

        self.assertEqual(metrics["microbatches"], 2)
        self.assertEqual(trainer.optimizer.step.item(), 1)


if __name__ == "__main__":
    unittest.main()
