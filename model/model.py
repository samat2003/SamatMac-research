"""SamatNext-520M full model assembly. All constants from config.py per SPEC.md."""

import mlx.core as mx
import mlx.nn as nn
from model.config import DEFAULT_CONFIG, SamatNextConfig
from model.delta_layer import GatedCausalAttentionLayer
from model.diff_attn_layer import DiffAttnLayer
from model.latent_tokens import LatentTokens
from model.memory_bus import MemoryBus
from model.mlp import SwiGLUMLP
from model.mtp_head import MTPHead


class SamatNext(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = []

        for layer_idx in range(config.n_layers):
            layer_type = config.get_layer_type(layer_idx)
            if layer_type == "gated_attention":
                self.layers.append(GatedCausalAttentionLayer(config, layer_idx))
            else:
                self.layers.append(DiffAttnLayer(config, layer_idx))

        self.mlps = [SwiGLUMLP(config) for _ in range(config.n_layers)]
        self.memory_bus = MemoryBus(config)
        self.latent_tokens = LatentTokens(config)
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_head = MTPHead(config)

    def __call__(
        self,
        input_ids: mx.array,
        targets: mx.array = None,
        loss_mask: mx.array = None,
    ) -> dict:
        x = self.embed(input_ids)
        x = x.astype(mx.bfloat16)
        x = self.latent_tokens.prepend(x)

        self.memory_bus.reset()
        for layer_idx in range(self.config.n_layers):
            layer = self.layers[layer_idx]
            mlp = self.mlps[layer_idx]
            layer_type = self.config.get_layer_type(layer_idx)
            if layer_type == "gated_attention":
                x = layer(x, memory_bus=self.memory_bus)
            else:
                x = layer(x)
            x = mlp(x)

        x = self.latent_tokens.strip(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}

        if targets is not None:
            if loss_mask is None:
                loss_mask = mx.ones(targets.shape, dtype=mx.bfloat16)

            token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
            mask = loss_mask.astype(token_losses.dtype)
            lm_loss = (token_losses * mask).sum() / mx.maximum(mask.sum(), 1.0)

            mtp_logits, mtp_confs = self.mtp_head(x)
            mtp_loss = self.mtp_head.compute_mtp_loss(
                mtp_logits,
                mtp_confs,
                targets,
                loss_mask,
            )
            loss = lm_loss + 0.3 * mtp_loss
            out["loss"] = loss
            out["lm_loss"] = lm_loss
            out["mtp_loss"] = mtp_loss

        return out

    def count_params(self) -> int:
        return count_params_exact(self)


def count_params_exact(model: SamatNext) -> int:
    total = 0
    cfg = model.config
    d = cfg.d_model
    v = cfg.vocab_size
    h = cfg.n_heads
    kv = cfg.n_kv_heads
    dh = cfg.d_head
    ff = cfg.d_ffn
    n = cfg.n_layers

    total += v * d
    for _ in range(cfg.num_gated_attention_layers):
        total += 5 * d * d + d + d
    for _ in range(cfg.num_diff_layers):
        total += (
            h * dh * 2 * d
            + kv * dh * 2 * d
            + kv * dh * d
            + d * d
            + d
            + 1
        )
    total += n * 3 * d * ff
    total += 3 * d * d + d
    total += cfg.n_latent_tokens * d
    total += d * d + cfg.mtp_heads * d * d + cfg.mtp_heads * d + d
    total += d
    total += 2 * v * d
    return total


def check_model(config: SamatNextConfig) -> None:
    model = SamatNext(config)
    print(f"Parameters: {model.count_params():,}")
    print(f"Parameters (exact): {count_params_exact(model):,}")
    input_ids = mx.zeros((1, 16), dtype=mx.int32)
    targets = mx.zeros((1, 16), dtype=mx.int32)
    loss_mask = mx.ones((1, 16), dtype=mx.bfloat16)
    out = model(input_ids, targets=targets, loss_mask=loss_mask)
    assert "logits" in out
    assert out["logits"].shape == (1, 16, config.vocab_size)
    assert "loss" in out
    assert out["loss"].shape == ()
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"LM loss: {out['lm_loss'].item():.4f}")
    print(f"MTP loss: {out['mtp_loss'].item():.4f}")
    print("SamatNext OK")


if __name__ == "__main__":
    check_model(DEFAULT_CONFIG)
