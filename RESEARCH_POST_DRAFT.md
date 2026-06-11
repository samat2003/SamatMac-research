# SamatNext: A Mac-Native Hybrid Sequence Model, and What Must Be Proven

## Abstract

SamatNext-520M is an experimental decoder-only language model implemented
directly in MLX for Apple Silicon. Its design alternates gated causal softmax
attention with Differential Attention, then adds a cross-layer memory bus, learned
prefix tokens, SwiGLU feed-forward layers, and a six-head multi-token prediction
objective.

The current implementation is best described as a hybrid attention model rather
than an attention replacement. Both mixer types construct full causal attention
matrices, so sequence mixing remains quadratic in context length. The individual
mechanisms also derive from prior work. The research question is therefore not
whether SamatNext has already replaced the Transformer, but whether this exact
composition improves quality, throughput, memory use, or sample efficiency on
Apple Silicon under controlled comparisons.

## Architecture

The configured model has:

- vocabulary size: 32,000
- model width: 1,024
- layers: 24
- nominal attention heads: 16
- grouped key/value heads: 4 in Differential Attention layers
- SwiGLU width: 4,096
- nominal context length: 4,096
- learned prefix tokens: 4
- multi-token prediction heads: 6

The input path is:

```text
token IDs
  -> token embedding
  -> prepend four learned prefix vectors
  -> 24 alternating mixer/MLP stages
       even stage: gated single-head causal attention + optional memory bus
       odd stage: grouped-query differential causal attention
       every stage: SwiGLU projection
  -> remove the four prefix positions
  -> RMSNorm
  -> vocabulary projection
```

### Even Layers: Gated Causal Attention

For hidden states `X`, the layer computes:

```text
G = sigmoid(X Wg)
Q = X Wq
K = normalize(X Wk)
V = X Wv
A = softmax(causal_mask(Q K^T / sqrt(d)))
Y = (G elementwise-multiplied with A) V
output = RMSNorm(X + Y Wo)
```

This is quadratic gated causal softmax attention. It is not a DeltaNet
recurrence: there is no fast-weight state update, delta-rule erase/write
operation, or chunkwise recurrent algorithm. `delta_proj` and
`delta_chunk_size` are defined but unused.

### Odd Layers: Differential Attention

Each odd layer builds two query/key systems:

```text
A1 = softmax(causal_mask(Q1 K1^T / sqrt(d_head)))
A2 = softmax(causal_mask(Q2 K2^T / sqrt(d_head)))
A  = A1 - clamp(lambda, 0, 1) A2
Y  = A V
output = RMSNorm(X + Y Wo)
```

Keys and values use grouped-query attention: four KV heads are repeated across
16 query heads. RoPE is applied to both query/key systems. The configured YaRN
parameters are not currently used.

This follows the central idea of Differential Transformer: subtracting one
softmax attention map from another to suppress shared attention noise.

### Cross-Layer Memory Bus

The final four even layers project their mixer output, calculate a causal prefix
mean at every position, and update an exponential moving state across layers:

```text
write[t] = mean(tanh(Y[0:t+1] Wwrite))
memory = write                         if memory is empty
memory = 0.9 memory + 0.1 write        otherwise
read = RMSNorm(Y + sigmoid(Y Wgate) * memory Wread)
```

This is a position-preserving causal communication channel across layers.
State at position `t` only depends on positions `0..t`, so it does not expose
future tokens to earlier predictions.

### Learned Prefix Tokens

Four trainable vectors are prepended to every sequence. Because the model is
causal, ordinary tokens can attend to these vectors, but the prefix vectors
cannot attend to the later input. They therefore behave as learned soft-prompt
tokens, not input-conditioned latent reasoning tokens.

### SwiGLU Stages

Every mixer is followed by:

```text
MLP(X) = Wdown(silu(Wgate X) elementwise-multiplied with Wup X)
```

The current model replaces the stream with the MLP output. It does not include
the usual residual connection around the feed-forward block.

### Multi-Token Prediction

Six auxiliary heads transform the final hidden state and predict future tokens
through vocabulary projections. A learned sigmoid confidence multiplies each
head's cross-entropy loss.

This is related to prior multi-token prediction work, but the current confidence
objective has a trivial solution: confidence can approach zero to reduce the
loss. A calibration target, regularizer, detached weighting rule, or explicit
accept/reject objective is needed.

## Complexity

Both sequence mixers currently materialize `sequence x sequence` score maps:

- even mixer: one full attention map
- odd mixer: two full attention maps for every query head
- training complexity: quadratic in sequence length
- autoregressive generation: no KV/recurrent cache, and the full prefix is
  recomputed after every generated token

Consequently, the current implementation cannot claim the linear-time or
constant-state advantages of Gated DeltaNet. A true delta-rule recurrence would
be the architectural change required to make that comparison.

## Parameter and Precision Audit

The corrected counter reports 520,930,316 parameters, matching the step-47,000
checkpoint. The embedding, language-model head, and MTP vocabulary projection
are three independently trained 32,000 by 1,024 arrays. The configuration now
states explicitly that these matrices are not tied.

The model should currently be called approximately `SamatNext-521M`, not
`SamatNext-520M`. Checkpoint weights are FP32, so the claim that weights,
activations, and optimizer state are BF16 everywhere is also unsupported.

## Training Audit

The workspace contains checkpoints through step 47,000, but no retained loss
curve, throughput report, validation result, HumanEval result, or baseline run.
The pretraining path now aligns each model position directly with the dataset's
already-shifted next-token target. Padding and FIM masks are applied to both the
main and multi-token losses. Gradients are averaged over eight microbatches,
clipped once, and followed by one optimizer update, giving the configured
effective batch size of 32.

Remaining implementation details prevent the existing checkpoints from
supporting a quality claim:
- The batch size selected by `SlidingMac` is not passed into the trainer.
- Checkpoints omit optimizer, scheduler, RNG, and step state. Resume restarts
  the learning-rate schedule and step counter.
- Evaluation intervals are configured but unused.
- The synthetic-data implementation uses local Qwen2.5-Coder-7B, while the
  specification says GPT-4o mini.

These are correctable engineering issues, but they must be fixed before using
training loss or benchmark scores as research evidence.

## What Is Potentially Novel

The individual components are established:

- Gated DeltaNet and hybrid DeltaNet/attention architectures
- Differential Attention
- learned latent or prefix vectors
- cross-layer memory mechanisms
- multi-token prediction
- SwiGLU, RoPE, grouped-query attention, and FIM training

The potentially novel contribution is the exact composition:

1. Alternating gated causal and differential-attention layers.
2. A causal cross-layer memory bus shared by selected recurrent layers.
3. Multi-token prediction specialized for a compact code model.
4. An MLX-first implementation and evaluation on constrained Apple unified
   memory.

Novel composition is a valid research contribution only if ablations show that
the combination produces gains beyond its parts.

## Claims That Can Be Made Now

- SamatNext is an experimental MLX-native hybrid decoder architecture.
- It combines differential attention, gated attention, learned prefix vectors,
  a cross-layer summary, SwiGLU blocks, and multi-token prediction.
- A substantial training run reached at least 47,000 saved steps on Apple
  Silicon.
- The project explores whether heterogeneous sequence mixers are useful for
  compact code models on consumer hardware.

## Claims That Require Evidence

Do not yet claim:

- superiority to Transformers or attention
- linear sequence complexity
- lower memory than a matched Transformer
- 360 million parameters
- BF16 weights and optimizer state
- 82-88% HumanEval
- faster inference or training
- improved reasoning from latent tokens
- sparse computation

## Required Baselines and Ablations

Train all models with the same tokenizer, data order, token budget, optimizer,
precision, width, parameter budget, and hardware:

1. Standard pre-norm Transformer with GQA.
2. Differential-Attention-only model.
3. Correct Gated-DeltaNet-only model.
4. Alternating Gated DeltaNet and standard attention.
5. Full SamatNext hybrid.

For the full model, remove one component at a time:

- memory bus
- learned prefix tokens
- multi-token prediction
- differential subtraction
- sparse/update gate
- alternating schedule

Report:

- validation cross-entropy and perplexity versus training tokens
- HumanEval, HumanEval+, and MBPP pass@1 with fixed decoding
- tokens per second at sequence lengths 256, 512, 1,024, 2,048, and 4,096
- peak unified memory
- prompt processing and token-generation latency
- parameter count and training FLOPs
- at least three seeds or uncertainty intervals

The central superiority claim should be narrow:

> At a matched parameter and training-token budget on Apple M4, the corrected
> SamatNext hybrid achieves X while the matched Transformer achieves Y, using Z
> memory and producing N tokens per second.

Without those matched numbers, "superior" is a hypothesis, not a result.

## Prior Work

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Yang et al., [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- Ye et al., [Differential Transformer](https://arxiv.org/abs/2410.05258)
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
