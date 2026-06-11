"""SwiGLU MLP block for SamatNext-520M. All constants from config.py per SPEC.md."""

import mlx.core as mx
import mlx.nn as nn

from model.config import DEFAULT_CONFIG, SamatNextConfig


class SwiGLUMLP(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.down_proj = nn.Linear(config.d_ffn, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def check_mlp(config: SamatNextConfig) -> None:
    mlp = SwiGLUMLP(config)
    x = mx.random.normal((1, 8, config.d_model)).astype(mx.bfloat16)
    y = mlp(x)
    assert y.shape == (1, 8, config.d_model)
    print("MLP OK")


if __name__ == "__main__":
    check_mlp(DEFAULT_CONFIG)
