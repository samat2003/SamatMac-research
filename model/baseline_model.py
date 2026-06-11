"""Standard decoder-only Baseline Transformer model."""

import math
import mlx.core as mx
import mlx.nn as nn

from model.config_120m import Baseline120MConfig
from model.mlp import SwiGLUMLP


class StandardAttention(nn.Module):
    def __init__(self, config: Baseline120MConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        self.rope = nn.RoPE(self.d_head, base=config.rope_base)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.d_head)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.n_kv_heads < self.n_heads:
            k = mx.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
            v = mx.repeat(v, self.n_heads // self.n_kv_heads, axis=1)

        if mask is not None:
            # mx.fast.scaled_dot_product_attention expects additive mask
            mask = mask.reshape(1, 1, L, L)
            
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0/self.scale, mask=mask
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: Baseline120MConfig):
        super().__init__()
        self.attn = StandardAttention(config)
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config)
        self.mlp_norm = nn.RMSNorm(config.d_model)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class BaselineModel(nn.Module):
    def __init__(self, config: Baseline120MConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.d_model)
        
        if config.tie_embeddings:
            self.lm_head = lambda x: mx.matmul(x, self.embed.weight.T)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        targets: mx.array = None,
        loss_mask: mx.array = None,
    ) -> dict:
        B, L = input_ids.shape
        x = self.embed(input_ids)
        x = x.astype(mx.bfloat16)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(L, x.dtype)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}

        if targets is not None:
            if loss_mask is None:
                loss_mask = mx.ones(targets.shape, dtype=mx.bfloat16)

            token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
            loss_m = loss_mask.astype(token_losses.dtype)
            lm_loss = (token_losses * loss_m).sum() / mx.maximum(loss_m.sum(), 1.0)
            
            out["loss"] = lm_loss
            out["lm_loss"] = lm_loss
            out["mtp_loss"] = mx.array(0.0) # Baseline has no MTP

        return out

    def count_params(self) -> int:
        return sum(v.size for k, v in tree_flatten(self.parameters()))

from mlx.utils import tree_flatten
