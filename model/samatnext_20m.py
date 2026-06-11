"""SamatNext architecture scaled down to 20M parameters."""

import mlx.core as mx
import mlx.nn as nn

from model.config_20m import SamatNext20MConfig
from model.delta_layer import GatedCausalAttentionLayer
from model.diff_attn_layer import DiffAttnLayer
from model.latent_tokens import LatentTokens
from model.memory_bus import MemoryBus
from model.mlp import SwiGLUMLP
from model.mtp_head import MTPHead
from mlx.utils import tree_flatten


class MTPHead20M(MTPHead):
    """MTP Head that respects the tie_embeddings flag."""
    def __init__(self, config: SamatNext20MConfig, embed_weight=None):
        super().__init__(config)
        self.tie_embeddings = config.tie_embeddings
        self._embed_weight = embed_weight

        if self.tie_embeddings:
            del self.vocab_proj

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = self.norm(x)
        all_logits = []
        all_confs = []

        for i in range(self.mtp_heads):
            h = nn.gelu(self.predictor(x) + self.predictor_offsets[i](x))
            if self.tie_embeddings and self._embed_weight is not None:
                logits_i = mx.matmul(h, self._embed_weight.T)
            else:
                logits_i = self.vocab_proj(h)
                
            conf_i = mx.sigmoid(self.confidence[i](x))
            all_logits.append(logits_i)
            all_confs.append(conf_i)

        logits = mx.stack(all_logits, axis=2)
        confidences = mx.stack(all_confs, axis=2)
        return logits, confidences


class SamatNext20M(nn.Module):
    def __init__(self, config: SamatNext20MConfig):
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
        self.latent_tokens = LatentTokens(config) if config.n_latent_tokens > 0 else None
        self.norm = nn.RMSNorm(config.d_model)
        
        if config.tie_embeddings:
            self.lm_head = lambda x: mx.matmul(x, self.embed.weight.T)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
        self.mtp_head = MTPHead20M(config, self.embed.weight if config.tie_embeddings else None)

    def __call__(
        self,
        input_ids: mx.array,
        targets: mx.array = None,
        loss_mask: mx.array = None,
    ) -> dict:
        x = self.embed(input_ids)
        x = x.astype(mx.bfloat16)
        
        if self.latent_tokens is not None:
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

        if self.latent_tokens is not None:
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
        return sum(v.size for k, v in tree_flatten(self.parameters()))
