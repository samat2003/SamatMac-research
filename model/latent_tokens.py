"""Latent reasoning tokens for SamatNext-520M. All constants from config.py per SPEC.md."""

import mlx.core as mx
import mlx.nn as nn

from model.config import DEFAULT_CONFIG, SamatNextConfig


class LatentTokens(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.n_latent = config.n_latent_tokens
        self.d_model = config.d_model
        self.latent = mx.zeros((config.n_latent_tokens, config.d_model), dtype=mx.bfloat16)

    def prepend(self, x: mx.array) -> mx.array:
        latent_batch = mx.broadcast_to(self.latent[None, :, :], (x.shape[0], self.n_latent, self.d_model))
        return mx.concatenate([latent_batch, x], axis=1)

    def strip(self, x: mx.array) -> mx.array:
        return x[:, self.n_latent:, :]

    def __call__(self, x: mx.array) -> mx.array:
        return self.prepend(x)


def check_latent_tokens(config: SamatNextConfig) -> None:
    lt = LatentTokens(config)
    x = mx.random.normal((2, 20, config.d_model)).astype(mx.bfloat16)
    out = lt.prepend(x)
    assert out.shape == (2, 24, config.d_model)
    stripped = lt.strip(out)
    assert stripped.shape == (2, 20, config.d_model)
    print("LatentTokens OK")


if __name__ == "__main__":
    check_latent_tokens(DEFAULT_CONFIG)
