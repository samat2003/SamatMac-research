"""Differential Attention layer with GQA for SamatNext-520M per SPEC.md."""

import math

import mlx.core as mx
import mlx.nn as nn

from model.config import DEFAULT_CONFIG, SamatNextConfig


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


class DiffAttnLayer(nn.Module):
    def __init__(self, config: SamatNextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.groups = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head * 2, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head * 2, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        self.lambda_param = mx.array(config.diff_attn_lambda_init, dtype=mx.bfloat16)
        self.norm = nn.RMSNorm(config.d_model)
        self.scale = math.sqrt(config.d_head)

        # Pre-compute RoPE sin/cos tables once (with headroom for latent tokens)
        rope_len = config.max_seq_len + 16
        freqs = 1.0 / (config.rope_base ** (mx.arange(0, self.d_head, 2) / self.d_head))
        positions = mx.arange(rope_len)
        sincos = mx.outer(positions, freqs)
        cos_table = mx.concatenate([mx.cos(sincos), mx.cos(sincos)], axis=-1)
        sin_table = mx.concatenate([mx.sin(sincos), mx.sin(sincos)], axis=-1)
        self._cos = cos_table.reshape(1, 1, rope_len, self.d_head)
        self._sin = sin_table.reshape(1, 1, rope_len, self.d_head)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape

        Q_full = self.q_proj(x)
        Q_full = Q_full.reshape(batch, seq_len, self.n_heads, self.d_head * 2)
        Q1, Q2 = mx.split(Q_full, 2, axis=-1)

        K_full = self.k_proj(x)
        K_full = K_full.reshape(batch, seq_len, self.n_kv_heads, self.d_head * 2)
        K1, K2 = mx.split(K_full, 2, axis=-1)

        V = self.v_proj(x)
        V = V.reshape(batch, seq_len, self.n_kv_heads, self.d_head)

        # Expand KV for GQA
        K1 = mx.repeat(K1, self.groups, axis=2)
        K2 = mx.repeat(K2, self.groups, axis=2)
        V = mx.repeat(V, self.groups, axis=2)

        Q1 = Q1.transpose(0, 2, 1, 3)
        Q2 = Q2.transpose(0, 2, 1, 3)
        K1 = K1.transpose(0, 2, 1, 3)
        K2 = K2.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Slice pre-computed RoPE tables for current seq_len
        cos = self._cos[:, :, :seq_len, :]
        sin = self._sin[:, :, :seq_len, :]

        Q1 = Q1 * cos + rotate_half(Q1) * sin
        Q2 = Q2 * cos + rotate_half(Q2) * sin
        K1 = K1 * cos + rotate_half(K1) * sin
        K2 = K2 * cos + rotate_half(K2) * sin

        mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bfloat16), k=1) * -1e9
        mask = mask.reshape(1, 1, seq_len, seq_len)
        
        out1 = mx.fast.scaled_dot_product_attention(Q1, K1, V, scale=1.0/self.scale, mask=mask)
        out2 = mx.fast.scaled_dot_product_attention(Q2, K2, V, scale=1.0/self.scale, mask=mask)
        
        out = out1 - mx.clip(self.lambda_param, 0.0, 1.0) * out2
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch, seq_len, self.n_heads * self.d_head)
        out = self.o_proj(out)
        return self.norm(x + out)


def check_diff_attn(config: SamatNextConfig) -> None:
    layer = DiffAttnLayer(config, layer_idx=1)
    x = mx.random.normal((2, 32, config.d_model)).astype(mx.bfloat16)
    y = layer(x)
    assert y.shape == (2, 32, config.d_model)
    print("DiffAttnLayer OK")


if __name__ == "__main__":
    check_diff_attn(DEFAULT_CONFIG)
