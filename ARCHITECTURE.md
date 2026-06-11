# SamatNext-520M Architecture

## Executive Summary

SamatNext-520M is a decoder-only autoregressive language model designed for
Python code generation and implemented natively in MLX for Apple Silicon.

Its defining architectural choice is an alternating sequence mixer:

```text
gated causal attention
        then
differential attention
        then
gated causal attention
        then
differential attention
        ...
```

There are 24 layers in total:

- 12 even-numbered gated causal attention layers
- 12 odd-numbered differential attention layers
- one SwiGLU feed-forward stage after every sequence mixer

The model also includes:

- four learned prefix vectors
- a causal cross-layer memory bus
- grouped-query attention in differential layers
- rotary position encoding in differential layers
- a six-head multi-token prediction objective
- fill-in-the-middle training for code completion

SamatNext is not an attention-free model and is not a DeltaNet. Both sequence
mixers materialize full causal attention matrices, giving the model quadratic
sequence complexity.

The model contains exactly 520,930,316 trainable elements.

## Configuration

| Property | Value |
| --- | ---: |
| Vocabulary | 32,000 |
| Model width | 1,024 |
| Layers | 24 |
| Gated attention layers | 12 |
| Differential attention layers | 12 |
| Query heads in differential attention | 16 |
| KV heads in differential attention | 4 |
| Head dimension | 64 |
| SwiGLU hidden width | 4,096 |
| Learned prefix vectors | 4 |
| Multi-token prediction heads | 6 |
| Maximum configured sequence | 4,096 |
| Dropout | 0 |
| Runtime activation type | BF16 |
| Parameter/checkpoint type | FP32 |

## End-to-End Data Flow

Given a batch of token IDs with shape:

```text
[batch, sequence]
```

the model performs the following computation:

```text
token IDs
   |
   v
32,000 x 1,024 token embedding
   |
   v
BF16 token representations
   |
   v
prepend four learned vectors
   |
   v
24 alternating mixer stages
   |
   +-- even layer: gated causal attention
   |
   +-- odd layer: differential attention
   |
   `-- every layer: SwiGLU transformation
   |
   v
remove the four learned prefix positions
   |
   v
final RMSNorm
   |
   v
independent 32,000-class language-model projection
   |
   v
next-token logits
```

During training, the final hidden states are also sent to the six-head
multi-token prediction module.

## Token Embedding

Each token ID is mapped to a 1,024-dimensional vector:

```text
X = Embedding(token_ids)
```

The embedding table has shape:

```text
[32,000, 1,024]
```

The embedding, primary language-model output, and multi-token output projection
are three independent matrices. They are not weight-tied.

After lookup, token representations are cast to BF16 for model computation.

## Learned Prefix Vectors

Four learned 1,024-dimensional vectors are prepended to every sequence:

```text
[p1, p2, p3, p4, token1, token2, ..., tokenN]
```

They provide trainable global starting states that every ordinary token can
access through causal attention.

Because they appear before the input under a causal mask, these prefix vectors
cannot read or summarize the input tokens. They are therefore most accurately
described as learned soft-prompt vectors, not input-conditioned reasoning
tokens.

The four prefix outputs are removed before token prediction, so the model emits
one logit vector for each original input position.

## Alternating Sequence Mixers

The model does not repeat one homogeneous Transformer block. Instead, it
alternates between two attention mechanisms with different inductive biases.

```text
Layer 0:  gated causal attention
Layer 1:  differential attention
Layer 2:  gated causal attention
Layer 3:  differential attention
...
Layer 22: gated causal attention
Layer 23: differential attention
```

The intended division of labor is:

- Gated causal attention performs direct content retrieval and controls how
  strongly each query position accepts retrieved context.
- Differential attention constructs two attention distributions and subtracts
  one from the other, attempting to cancel context shared by both maps.
- Alternation lets later layers repeatedly switch between context accumulation
  and context contrast.

This division of labor is a hypothesis that must be demonstrated through
matched baselines and ablations.

## Gated Causal Attention

### Projections

For an input tensor `X` of shape `[B, S, 1024]`, the layer computes:

```text
Q = X Wq
K = X Wk
V = X Wv
G = sigmoid(X Wg)
```

The query, key, and value projections all have width 1,024. The gate produces
one scalar for each batch item and sequence position:

```text
G shape = [B, S, 1]
```

Despite the configuration containing 16 heads, this layer does not split its
Q, K, and V tensors into heads. It behaves as one wide 1,024-dimensional
attention system.

### Key Normalization

Each key vector is normalized by its L2 norm:

```text
K_t = K_t / sqrt(sum(K_t^2) + epsilon)
```

This constrains key magnitude, making query direction and magnitude more
important than arbitrary key scaling.

### Causal Attention

The layer computes the full score matrix:

```text
scores = Q K^T / sqrt(1024)
```

Future positions are replaced with a large negative value before softmax:

```text
A = softmax(causal_mask(scores))
```

The output gate is then applied:

```text
A_gated[b, s, t] = G[b, s] * A[b, s, t]
Y = A_gated V
```

The gate does not select individual keys. It scales the complete retrieved
context for each query position. It is dense and continuous, not sparse
routing.

The mixed output is clipped to `[-10, 10]`, projected, added to the input, and
normalized:

```text
output = RMSNorm(X + Wo clip(Y, -10, 10))
```

### Complexity

The attention matrix has shape:

```text
[batch, sequence, sequence]
```

Therefore:

- attention compute is `O(S^2 * d)`
- score memory is `O(S^2)`
- this layer is quadratic, not linear or recurrent

The source retains an unused `delta_proj` matrix and a reserved chunk-size
configuration. These do not currently participate in the forward pass.

## Differential Attention

Differential attention creates two independent attention maps and subtracts the
second from the first.

### Head Structure

The layer uses:

- 16 query heads
- 4 key/value heads
- head dimension 64
- 4 query heads per KV group

This is grouped-query attention. It reduces key/value projection size compared
with assigning independent K and V heads to all 16 query heads.

### Dual Query and Key Systems

The query projection produces two vectors for each query head:

```text
Q -> [Q1, Q2]
```

The key projection similarly produces:

```text
K -> [K1, K2]
```

One value representation `V` is shared by both systems.

After reshaping:

```text
Q1, Q2: [B, 16, S, 64]
K1, K2: [B, 16, S, 64]
V:      [B, 16, S, 64]
```

The four physical KV heads are repeated across their query groups before
attention is computed.

### Rotary Position Encoding

RoPE is applied to both query/key pairs:

```text
(Q1, K1) and (Q2, K2)
```

RoPE rotates pairs of hidden dimensions according to token position, allowing
dot products to encode relative position.

The configured YaRN values are not currently used. RoPE is also not applied in
the gated causal attention layers.

### Differential Map

Two causal softmax maps are produced:

```text
A1 = softmax(causal_mask(Q1 K1^T / sqrt(64)))
A2 = softmax(causal_mask(Q2 K2^T / sqrt(64)))
```

They are combined using one learned scalar per layer:

```text
A = A1 - clamp(lambda, 0, 1) * A2
```

`lambda` starts at `0.8`.

The intuition is noise cancellation. If both maps attend strongly to generic or
irrelevant context, subtraction can suppress that shared component. Context
that is emphasized by the first map but not the second remains strong.

The resulting weights are not a probability distribution:

- entries can be negative
- rows do not have to sum to one
- the second map can actively cancel value contributions

The layer then computes:

```text
Y = A V
output = RMSNorm(X + Wo Y)
```

### Complexity

Differential attention forms two full per-head score matrices:

```text
[B, 16, S, S]
```

It remains quadratic and is more attention-map-intensive than the even-layer
mixer.

## Causal Cross-Layer Memory Bus

The memory bus provides an additional route between selected gated-attention
layers.

It is reset at the beginning of every model invocation, so it does not persist
between independent samples or generation calls.

### Writers

Only gated-attention layers 16, 18, 20, and 22 write to the bus.

For mixer output `Y`, a writer calculates:

```text
W = tanh(Y Wwrite)
```

For every position `t`, it computes a causal prefix mean:

```text
P_t = mean(W_0, W_1, ..., W_t)
```

This is implemented with cumulative sums. Position `t` can only contain
information from positions at or before `t`, preserving autoregressive
causality.

Across writer layers, the state is updated as:

```text
M = P                           for the first writer
M = 0.9 * M + 0.1 * P           for later writers
```

This gives the memory bus depth-wise persistence while keeping a separate
causal state for every sequence position.

### Readers

Once the bus has a state, each gated-attention layer reading it computes:

```text
gate = sigmoid(Y Wgate)
read = M Wread
output = RMSNorm(Y + gate * read)
```

The read gate has 1,024 channels at every position. Different representation
dimensions can therefore accept different amounts of memory.

The writer immediately reads the state it has just updated.

### Interpretation

Ordinary residual streams pass information from one layer to the next. The bus
adds a second depth-wise path that summarizes causal prefixes and can be
reintroduced under learned channel-wise control.

It is not:

- persistent memory across documents
- an external database
- a recurrent inference cache
- a replacement for the attention matrix

## SwiGLU Feed-Forward Stage

Every mixer is followed by a SwiGLU transformation:

```text
gate = SiLU(X Wgate)
value = X Wup
output = Wdown(gate * value)
```

Its dimensions are:

```text
1,024 -> 4,096
1,024 -> 4,096
4,096 -> 1,024
```

This is the largest parameter component of the model. Across 24 layers, the
SwiGLU matrices contain roughly 302 million elements.

Unlike a conventional Transformer feed-forward sublayer, the current code does
not add the SwiGLU output back to its input and does not normalize around this
stage:

```text
X = MLP(X)
```

rather than:

```text
X = X + MLP(norm(X))
```

This makes the feed-forward stage a full replacement transformation. It is a
major architectural difference and should be included in ablations because it
changes optimization and information preservation behavior.

## Output Heads

### Primary Language-Model Head

After the final layer:

```text
H = RMSNorm(H)
logits = H Wlm
```

`Wlm` has shape `[32,000, 1,024]`.

The logits at position `t` predict the dataset target already shifted to token
`t + 1`. The model does not shift targets a second time.

### Six-Head Multi-Token Prediction

The auxiliary module asks each final hidden state to predict targets at
multiple future offsets.

First:

```text
Z = RMSNorm(H)
shared = Z Wshared
```

For head `i`:

```text
H_i = GELU(shared + Z Woffset_i)
MTP_logits_i = H_i Wmtp_vocab
confidence_i = sigmoid(Z Wconfidence_i)
```

The six heads use independent offset transformations and confidence projections
but share one MTP vocabulary matrix.

Head zero predicts the same next-token target as the primary LM head. Later
heads predict progressively farther targets by aligning earlier hidden
positions with later target positions.

The auxiliary objective is:

```text
total_loss = next_token_loss + 0.3 * mean(MTP_head_losses)
```

Padding and FIM masks apply to both objectives.

The confidence currently multiplies cross-entropy directly. This design should
be monitored because decreasing confidence can decrease the auxiliary loss.
A calibration term or constrained confidence objective may be required to
prevent trivial confidence collapse.

## Training Data and Fill-in-the-Middle

The model is specialized for Python.

For ordinary autoregressive samples:

```text
input:  [BOS, t1, t2, t3]
target: [t1,  t2, t3, t4]
```

Only real target positions receive a loss weight. Padding positions are masked
out.

For 50% of samples, source code is divided into prefix, middle, and suffix:

```text
<fim_prefix> prefix
<fim_suffix> suffix
<fim_middle> middle
<eos>
```

The causal model sees the prefix and suffix before generating the missing
middle. The training mask applies loss only where the model predicts middle
tokens. Context-control tokens, prefix, suffix, EOS, and padding do not
contribute to the FIM loss.

This teaches insertion and code-completion behavior without introducing
bidirectional attention.

## Optimization

Training uses:

- microbatch size 4
- eight-microbatch gradient accumulation
- effective batch size 32
- AdamW
- learning rate `3e-4`
- betas `(0.9, 0.95)`
- weight decay `0.1`
- 500 warmup steps
- cosine decay
- global gradient clipping at `1.0`

For every optimizer step:

1. Eight microbatch gradients are computed.
2. Gradients are summed.
3. The sum is divided by eight.
4. The averaged global gradient is clipped once.
5. AdamW performs one update.

This is mathematically different from clipping each microbatch independently
and then averaging.

## Parameter Distribution

The exact total is:

```text
520,930,316 trainable elements
```

The largest groups are approximately:

| Component | Elements |
| --- | ---: |
| 24 SwiGLU stages | 301,989,888 |
| 24 sequence mixers | 110,137,356 |
| MTP module | 40,115,200 |
| Token embedding | 32,768,000 |
| LM vocabulary head | 32,768,000 |
| Memory bus | 3,146,752 |
| Prefix vectors and final norm | small remainder |

The independent MTP vocabulary projection is included within the MTP count.

FP32 parameters require approximately 2.08 GB. Parameters, two AdamW moment
buffers, and gradients require approximately 8.33 GB before activation memory
and macOS memory use.

## Computational Characteristics

### Training

Both attention mechanisms are quadratic:

```text
compute grows approximately with sequence_length^2
```

Doubling sequence length roughly quadruples attention-map work and storage,
ignoring projection and MLP costs.

The configured `4 x 4096` microbatch has not yet been demonstrated to fit in
16 GB unified memory. Sequence length and batch size must be measured before
the full pretraining run.

### Inference

Generation currently recomputes the complete sequence for every new token.
There is no KV cache or recurrent cache.

Consequently, current autoregressive inference is substantially less efficient
than a production Transformer implementation with cached keys and values.

## How It Differs From a Standard Transformer

| Standard decoder Transformer | SamatNext-520M |
| --- | --- |
| Same multi-head attention in every layer | Alternates two attention mechanisms |
| One softmax map per head | Odd layers subtract two softmax maps |
| Usually RoPE in every attention layer | RoPE only in differential layers |
| No explicit shared cross-layer memory | Final gated layers use a causal memory bus |
| Usually residual MLP sublayer | MLP output replaces the stream |
| Usually next-token loss only | Next-token plus six-head future-token loss |
| Optional prompt tokens | Four learned prefix vectors are always present |
| Often tied embedding/output weights | Three vocabulary matrices are independent |

## What the Architecture Is Trying to Achieve

The architecture combines several biases:

1. **Direct retrieval:** gated causal attention retrieves weighted combinations
   of earlier values.
2. **Noise cancellation:** differential attention subtracts a competing context
   map.
3. **Depth-wise memory:** the causal bus carries prefix summaries across final
   gated layers.
4. **Global learned priors:** prefix vectors provide trainable starting context.
5. **Future-aware representations:** multi-token prediction pressures hidden
   states to encode more than the immediate next token.
6. **Code insertion skill:** FIM training teaches generation between known
   prefix and suffix code.

These are architectural motivations, not established performance results.
Claims of improved quality, speed, memory efficiency, or reasoning require
matched Transformer baselines and component ablations.

## Accurate One-Paragraph Explanation

SamatNext-520M is a 24-layer, decoder-only Python code model built in MLX. It
alternates a wide gated causal softmax attention layer with a 16-head
differential-attention layer that subtracts two causal attention maps. Four
learned soft-prompt vectors are prepended to every sequence, and the last four
gated-attention layers communicate through a causal cross-layer prefix-memory
bus. Every mixer is followed by a 4,096-wide SwiGLU transformation, and final
hidden states train against both ordinary next-token prediction and six
confidence-weighted future-token objectives. The model has 520,930,316
parameters, uses BF16 activations and FP32 parameter storage, and remains
quadratic in sequence length because both mixers construct full attention
matrices.

## Claims to Avoid

Do not describe the current model as:

- a DeltaNet
- attention-free
- linear-time
- sparse attention
- recurrent
- a proven Transformer replacement
- a model with input-conditioned latent reasoning tokens
- a model with persistent memory
- inference-optimized

The accurate description is:

> A Mac-native, quadratic hybrid attention model combining gated causal
> attention, differential attention, causal cross-layer prefix memory,
> learned soft prompts, SwiGLU transformations, and multi-token prediction.

---

## 120M Validation Experiment Specification

To validate the SamatNext architecture empirically, we run a parameter-matched scaling experiment at the 120M (117M) parameter scale. This model size is chosen to enable rapid iterations on Apple Silicon (M4 iMac 16GB) under identical training budgets, compared against a standard decoder-only Transformer.

### Model Parameter Budgets

| Property | Baseline-120M | SamatNext-120M |
| :--- | :---: | :---: |
| Trainable Parameters | 116,583,168 (116.6M) | 116,883,461 (116.9M) |
| Model Dimension ($d_{\text{model}}$) | 768 | 768 |
| Layers ($L$) | 10 | 10 (5 Gated Causal, 5 Differential) |
| Attention Heads ($H_q$) | 12 | 12 |
| KV Heads ($H_{kv}$) | 4 | 4 |
| Head Dimension ($d_{\text{head}}$) | 64 | 64 |
| FFN Inner Width ($d_{\text{ffn}}$) | 3310 (wider to match params) | 2700 |
| Memory Bus Layers | - | 2 (layers 8 and 10 write; 8 and 10 read) |
| Latent Tokens | - | 4 |
| MTP Predictor Heads | - | 2 |
| Vocabulary Size ($V$) | 32,000 | 32,000 |
| Maximum Sequence Length | 512 | 512 |

### Dataset Details & Processing Pipeline

The pre-training and fine-tuning datasets are processed under strict protocols to prevent data leakage and maximize validation rigor:

1. **Pre-training Dataset: GitHub Clean Code (Python)**
   * **Source**: `codeparrot/github-code-clean` (Python subset) & `bigcode/the-stack-v2-train-smol-ids`.
   * **Volume**: 100,000 samples (~8,000,000 tokens).
   * **Filtering & Deduplication**: Exact-match deduplication is enforced by indexing the first 50 characters of each file combined with size-bucket sorting. Files containing standard boilerplate, autogenerated headers, or minified text are automatically filtered.
   * **Purpose**: Teach the models fundamental Python syntax, imports, standard library usage, and mathematical conventions from scratch.

2. **Fine-tuning Dataset: Python-Codes-25k**
   * **Source**: `flytech/python-codes-25k`.
   * **Volume**: ~49,627 samples (~25.8 MB of JSONL containing instruction+code pairs).
   * **Purpose**: Align the pre-trained models to follow code-completion prefixes and solve simple instruction-following code tasks.

### Tokenization & Vocabulary
We train a custom BPE (Byte Pair Encoding) tokenizer wrapper using the HuggingFace `tokenizers` library with a vocabulary size of $V = 32,000$. Special control tokens are reserved for syntax parsing and Fill-In-The-Middle training:
* `<bos>` (ID: 0): Beginning of sequence.
* `<eos>` (ID: 1): End of sequence.
* `<pad>` (ID: 2): Padding.
* `<fim_prefix>` (ID: 3): Boundary token indicating FIM prefix start.
* `<fim_suffix>` (ID: 4): Boundary token indicating FIM suffix start.
* `<fim_middle>` (ID: 5): Boundary token indicating FIM middle start.

### Fill-In-The-Middle (FIM) Transformation
To enable bidirectional prefix/suffix reasoning without bidirectional attention, 50% of the training samples are transformed on-the-fly using the Fill-in-the-Middle (FIM) protocol (joint autoregressive prefix-suffix training):
1. A document $D$ is split into three chunks: $D = (P, M, S)$ (prefix, middle, suffix).
2. The sequence is formatted as:
   $$\text{Input} = \langle\text{fim\_prefix}\rangle P \langle\text{fim\_suffix}\rangle S \langle\text{fim\_middle}\rangle M \langle\text{eos}\rangle$$
3. Loss calculation is masked to only include the token positions within $M$.

### Validation Benchmark Metrics

The 120M models are evaluated on two code completion suites to measure functional correctness:
1. **Token-Level Code Completion (15 tasks)**: Evaluates the model's exact text matches for short continuations (e.g. `+ b` or `, name, age):`).
2. **Execution-Based Functional Correctness (80 tasks, 8 categories)**: Measures **pass@1** correctness using greedy decoding ($T=0$). For each task, the model completes a logical continuation prompt, which is compiled and executed alongside unit test assertions in an isolated environment.

