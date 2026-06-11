"""Quadratic gated causal attention for SamatNext-520M per SPEC.md."""

import math
import mlx.core as mx
import mlx.nn as nn

from model.config import DEFAULT_CONFIG, SamatNextConfig


class GatedCausalAttentionLayer(nn.Module):
    def __init__(self, config: SamatNextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = 1  # Single-head for gated attention
        self.d_head = config.d_model
        self.chunk_size = config.delta_chunk_size
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.delta_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.sparse_gate = nn.Linear(self.d_model, 1, bias=False)
        self.norm = nn.RMSNorm(self.d_model)
        self.scale = 1.0 / math.sqrt(self.d_model)

    def __call__(self, x: mx.array, memory_bus=None) -> mx.array:
        batch, seq_len, _ = x.shape

        g = mx.sigmoid(self.sparse_gate(x))
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        mask = mx.triu(mx.ones((seq_len, seq_len), dtype=x.dtype), k=1) * -1e9
        mask = mask.reshape(1, 1, seq_len, seq_len)
        
        # Reshape for SDPA: [B, 1, L, D]
        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.squeeze(1) * g
        out = mx.clip(out, -10.0, 10.0)

        if memory_bus is not None:
            is_writer = self.layer_idx >= self.config.memory_bus_start_layer
            out = memory_bus(out, do_write=is_writer)

        out = self.o_proj(out)
        return self.norm(x + out)


def check_delta_layer(config: SamatNextConfig) -> None:
    from model.memory_bus import MemoryBus

    layer = GatedCausalAttentionLayer(config, layer_idx=0)
    x = mx.random.normal((2, 64, config.d_model)).astype(mx.bfloat16)
    y = layer(x, memory_bus=None)
    assert y.shape == (2, 64, config.d_model)
    bus = MemoryBus(config)
    y = layer(x, memory_bus=bus)
    assert y.shape == (2, 64, config.d_model)
    print("GatedCausalAttentionLayer OK")


# Compatibility alias for checkpoints and older imports. This layer is not a
# DeltaNet recurrence; it materializes a full causal softmax attention matrix.
GatedDeltaNetLayer = GatedCausalAttentionLayer


if __name__ == "__main__":
    check_delta_layer(DEFAULT_CONFIG)
