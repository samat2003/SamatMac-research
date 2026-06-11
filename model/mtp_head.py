"""Confidence-gated multi-token prediction head for SamatNext-520M per SPEC.md."""

import mlx.core as mx
import mlx.nn as nn

from model.config import DEFAULT_CONFIG, SamatNextConfig


class MTPHead(nn.Module):
    def __init__(self, config: SamatNextConfig):
        super().__init__()
        self.config = config
        self.mtp_heads = config.mtp_heads
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.predictor = nn.Linear(config.d_model, config.d_model, bias=False)
        self.predictor_offsets = [
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.mtp_heads)
        ]
        self.confidence = [
            nn.Linear(config.d_model, 1, bias=False)
            for _ in range(config.mtp_heads)
        ]
        self.norm = nn.RMSNorm(config.d_model)
        self.vocab_proj = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = self.norm(x)
        all_logits = []
        all_confs = []

        for i in range(self.mtp_heads):
            h = nn.gelu(self.predictor(x) + self.predictor_offsets[i](x))
            logits_i = self.vocab_proj(h)
            conf_i = mx.sigmoid(self.confidence[i](x))
            all_logits.append(logits_i)
            all_confs.append(conf_i)

        logits = mx.stack(all_logits, axis=2)
        confidences = mx.stack(all_confs, axis=2)
        return logits, confidences

    def compute_mtp_loss(
        self,
        logits: mx.array,
        confidences: mx.array,
        targets: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        losses = []

        for i in range(self.mtp_heads):
            if i == 0:
                target_i = targets
                logits_i = logits[:, :, i, :]
                conf_i = confidences[:, :, i, :]
                mask_i = loss_mask
            else:
                target_i = targets[:, i:]
                logits_i = logits[:, :-i, i, :]
                conf_i = confidences[:, :-i, i, :]
                mask_i = loss_mask[:, i:]

            ce_i = nn.losses.cross_entropy(logits_i, target_i, reduction="none")
            mask_i = mask_i.astype(ce_i.dtype)
            weights = conf_i.squeeze(-1) * mask_i
            loss_i = (ce_i * weights).sum() / mx.maximum(mask_i.sum(), 1.0)
            losses.append(loss_i)

        return mx.stack(losses).mean()


def check_mtp_head(config: SamatNextConfig) -> None:
    head = MTPHead(config)
    x = mx.random.normal((2, 16, config.d_model)).astype(mx.bfloat16)
    targets = mx.zeros((2, 16), dtype=mx.int32)
    logits, confs = head(x)
    assert logits.shape == (2, 16, config.mtp_heads, config.vocab_size)
    assert confs.shape == (2, 16, config.mtp_heads, 1)
    loss_mask = mx.ones(targets.shape, dtype=mx.bfloat16)
    loss = head.compute_mtp_loss(logits, confs, targets, loss_mask)
    assert loss.shape == ()
    print("MTPHead OK")


if __name__ == "__main__":
    check_mtp_head(DEFAULT_CONFIG)
