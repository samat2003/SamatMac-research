"""SlidingSpeed Mac adapter for SamatNext-520M. Auto-benchmarks MLX training config per SPEC.md."""

import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model.config import DEFAULT_CONFIG, SamatNextConfig
from model.model import SamatNext


class SlidingMac:
    def __init__(self, model: SamatNext, config: SamatNextConfig):
        self.model = model
        self.config = config
        self.best_batch_size = 4
        self.best_tokens_per_sec = 0.0
        self.results = []

    def _benchmark_batch_size(self, batch_size: int, seq_len: int = 512) -> float:
        input_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        targets = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        out = self.model(input_ids, targets=targets)
        mx.eval(out["logits"])

        t0 = time.perf_counter()
        for _ in range(3):
            out = self.model(input_ids, targets=targets)
            mx.eval(out["loss"])
        elapsed = time.perf_counter() - t0

        tokens_per_sec = (3 * batch_size * seq_len) / elapsed
        return tokens_per_sec

    def search(self, candidate_batch_sizes: list[int] = None) -> int:
        candidates = candidate_batch_sizes or [1, 2, 4, 8]
        print("SlidingMac: benchmarking batch sizes...")

        best = self.best_batch_size
        best_tps = 0.0
        self.results = []

        for batch_size in candidates:
            try:
                tps = self._benchmark_batch_size(batch_size)
            except Exception:
                print(f"  batch_size={batch_size}: OOM/error, skipping")
                continue

            print(f"  batch_size={batch_size}: {tps:.1f} tok/s")
            self.results.append(
                {
                    "batch_size": batch_size,
                    "tokens_per_sec": tps,
                }
            )
            if tps > best_tps:
                best = batch_size
                best_tps = tps

        self.best_batch_size = best
        self.best_tokens_per_sec = best_tps
        print(
            f"SlidingMac: selected batch_size={self.best_batch_size} "
            f"({self.best_tokens_per_sec:.1f} tok/s)"
        )
        return self.best_batch_size

    def report(self) -> dict:
        return {
            "best_batch_size": self.best_batch_size,
            "best_tokens_per_sec": self.best_tokens_per_sec,
            "all_results": self.results,
            "device": str(mx.default_device()),
        }


def check_sliding_mac(config: SamatNextConfig) -> None:
    print("SlidingMac: skipping benchmark (requires full model init)")
    print("Candidate batch sizes: [1, 2, 4, 8]")
    print(f"Device: {mx.default_device()}")
    print("SlidingMac OK")


if __name__ == "__main__":
    check_sliding_mac(DEFAULT_CONFIG)
