"""Causal cross-layer memory bus for SamatNext-520M per SPEC.md."""

import mlx.core as mx
import mlx.nn as nn

from model.config import DEFAULT_CONFIG, SamatNextConfig


class MemoryBus(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.write_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.read_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = nn.RMSNorm(config.d_model)
        self._bus_state = None

    def reset(self) -> None:
        self._bus_state = None

    def write(self, x: mx.array) -> None:
        w = mx.tanh(self.write_proj(x))
        prefix_sum = mx.cumsum(w, axis=1)
        prefix_count = mx.arange(1, x.shape[1] + 1, dtype=x.dtype)
        w = prefix_sum / prefix_count.reshape(1, -1, 1)
        if self._bus_state is None:
            self._bus_state = w
        else:
            self._bus_state = 0.9 * self._bus_state + 0.1 * w

    def read(self, x: mx.array) -> mx.array:
        if self._bus_state is None:
            return x
        g = mx.sigmoid(self.gate(x))
        r = self.read_proj(self._bus_state)
        output = self.norm(x + g * r)
        return output

    def __call__(self, x: mx.array, do_write: bool = False) -> mx.array:
        if do_write:
            self.write(x)
        return self.read(x)


def check_memory_bus(config: SamatNextConfig) -> None:
    bus = MemoryBus(config)
    x1 = mx.random.normal((2, 16, config.d_model)).astype(mx.bfloat16)
    x2 = mx.random.normal((2, 16, config.d_model)).astype(mx.bfloat16)
    y1 = bus(x1, do_write=True)
    assert y1.shape == (2, 16, config.d_model)
    y2 = bus(x2, do_write=False)
    assert y2.shape == (2, 16, config.d_model)
    bus.reset()
    assert bus._bus_state is None
    print("MemoryBus OK")


if __name__ == "__main__":
    check_memory_bus(DEFAULT_CONFIG)
