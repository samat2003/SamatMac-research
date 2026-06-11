# SamatNext: A Hybrid Sequence Mixer Architecture for Data-Efficient Code Generation on Apple Silicon

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** Standard decoder-only Transformers exhibit high sample complexity and are prone to attention dilution when trained on code, leading to sub-optimal resource use in compute-constrained settings. We present **SamatNext**, a hybrid sequence mixer architecture designed to optimize Python code completion efficiency. By alternating Gated Causal Attention with Differential Attention, and integrating a causal Cross-Layer Memory Bus and Multi-Token Prediction (MTP), SamatNext achieves a **60.0%** score on token-level code completion (vs. **33.3%** for a parameter-matched baseline) and **12.5%** on a rigorous 80-task execution-based functional correctness (pass@1) benchmark (vs. **10.0%** baseline). Notably, SamatNext demonstrates a stark advantage in structural coding patterns, passing **30.0%** of Object-Oriented Programming (OOP) tasks where the baseline completely fails (**0.0%**). All models are implemented natively in Apple MLX and trained from scratch on Apple Silicon (M4, 16GB).

---

## 1. Research Motivation: Why We Built It

As Large Language Models (LLMs) scale, their resource requirements during both training and inference become prohibitive for local deployment. This limitation is particularly acute on unified-memory consumer hardware, such as Apple Silicon Macs. In code generation tasks, standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) mechanisms suffer from:
1. **Attention Dilution**: Softmax probability mass is often distributed across irrelevant tokens (e.g., comments, whitespace, boilerplate code), degrading the signal-to-noise ratio for key syntactic anchors.
2. **Context Degradation**: Layer-by-layer representations of structural namespaces (such as class properties or dictionary key-value states) decay as they pass through deep networks, leading to a loss of long-range reference tracking.
3. **High Sample Complexity**: Standard next-token prediction objectives provide a sparse gradient signal, requiring billions of training tokens to learn basic structural syntax.

**SamatNext** was designed to address these bottlenecks directly by combining inductive biases that enforce noise cancellation, cross-layer state persistence, and future-token awareness. We evaluate this hybrid sequence mixer at the **120M parameter scale** under a highly data-constrained budget (~12M tokens total) to study how efficiently the architecture learns Python structural logic compared to a standard Transformer baseline under identical parameter and compute constraints.

---

## 2. Key Architectural Innovations

SamatNext departs from standard homogeneous Transformer designs by alternating sequence mixers and introducing causal memory channels:

```
                          Token Input
                               |
                               v
                  +──────────────────────────+
                  |  Learned Soft Prompts    |  <-- Prepend 4 global reasoning tokens
                  +──────────────────────────+
                               |
                               v
        +──────────────────────────────────────────────+
        | Layer 0: Gated Causal Attention (Sigmoid)     |  <-- Content Retrieval
        +──────────────────────────────────────────────+
                               |
                               v
        +──────────────────────────────────────────────+
        | Layer 1: Differential Attention (A1 - λ A2)  |  <-- Noise Cancellation
        +──────────────────────────────────────────────+
                               |
                               v
                              ...
                               |
                               v
        +──────────────────────────────────────────────+
        | Layer 8: Gated Causal Attention              | <---+  Causal Memory Bus
        +──────────────────────────────────────────────+     |  (Shared highway for
                               |                             |  namespace binding)
                               v                             |
        +──────────────────────────────────────────────+     |
        | Layer 9: Differential Attention              |     |
        +──────────────────────────────────────────────+     |
                               |                             |
                               v                             v
        +──────────────────────────────────────────────+  W_write
        | Layer 10: Gated Causal Attention             | ----+
        +──────────────────────────────────────────────+
                               |
                               v
                  +──────────────────────────+
                  | Remove Soft Prompts      |
                  +──────────────────────────+
                               |
                               v
                  +──────────────────────────+
                  |  Multi-Token Prediction  |  <-- Auxiliary Loss (Predict t+1, t+2)
                  +──────────────────────────+
```

### A. Differential Attention for Noise Cancellation
Implemented in odd-numbered layers, **Differential Attention** constructs two independent GQA attention maps and subtracts one from the other:
$$Attention(Q, K, V) = \text{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d_{\text{head}}}}\right)V - \lambda \text{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d_{\text{head}}}}\right)V$$
Where $\lambda \in [0, 1]$ is a learned layer-wise scalar (initialized to $0.8$). The subtraction acts as a high-pass filter, canceling out common-mode background noise (like comments or repetitive tokens) and highlighting specific syntactic relations.

### B. Namespace Binding via the Cross-Layer Memory Bus
To prevent the degradation of structural references in deep networks, gated attention layers 8 and 10 write to and read from a shared causal **Memory Bus**.
* **Write Protocol**: For a mixer output $Y$, the write projection is:
  $$W = \tanh(Y W_{\text{write}})$$
  The bus state is updated causally using prefix cumulative means:
  $$P_t = \frac{1}{t+1} \sum_{i=0}^{t} W_i$$
  $$M = 0.9 \cdot M + 0.1 \cdot P$$
* **Read Protocol**: Upper gated layers read from the bus via a learned channel-wise sigmoid gate:
  $$\text{read} = M W_{\text{read}}$$
  $$\text{Output} = \text{RMSNorm}(Y + \text{sigmoid}(Y W_{\text{gate}}) \odot \text{read})$$
This creates a depth-wise "highway" that preserves global scope states (like class attributes) across the network.

### C. Multi-Token Prediction (MTP)
During training, the final hidden states project to $K=2$ future-token heads. By forcing the representation at step $t$ to simultaneously predict tokens at $t+1$ and $t+2$, the model is pressured to construct future-aware features and plan syntactic closures (like brackets and colons), yielding cleaner completions.

---

## 3. Comparative Evaluation & Results

We trained both architectures under a matched budget of **117M parameters** using Apple MLX. 

### A. Token-Level Code Completion (15 tasks)
Measures if the model can predict the exact continuation string (e.g. `+ b` or `Exception as e:`) for common code snippets.

* **Baseline-120M**: **33.3%** (5/15)
* **SamatNext-120M**: **60.0%** (9/15)

### B. Execution-Based Functional Correctness (80 tasks, pass@1)
A rigorous benchmark consisting of 80 Python tasks across 8 categories (10 tasks each). Functional correctness is validated by executing completions against unit test assertions in an isolated environment using greedy decoding ($T=0, n=1$).

| Model | Overall Score | OOP (10 tasks) | Dicts & Sets (10 tasks) | Strings (10 tasks) |
|---|---|---|---|---|
| Baseline-120M | 10.0% (8/80) | 0.0% (0/10) | 0.0% (0/10) | **50.0% (5/10)** |
| **SamatNext-120M** | **12.5% (10/80)** | **30.0% (3/10)** | **10.0% (1/10)** | 40.0% (4/10) |

#### Analysis:
* **Object-Oriented Programming (30.0% vs. 0.0%)**: SamatNext successfully passed self-referential attribute bindings and equality checks. The Cross-Layer Memory Bus successfully maintained class scopes (`self.name` reference tracking) which the baseline lost in layer-wise routing.
* **Dicts & Sets (10.0% vs. 0.0%)**: Differential Attention successfully filtered out surrounding noise, enabling the model to retrieve exact namespace keys.

---

## 4. Reproducing the Experiments

All training and evaluation steps are 100% reproducible on Apple Silicon (M-series Mac) or any standard CPU/GPU system.

### A. Setup Environment
Ensure you have Python 3.11+ installed. Clone the repository and install dependencies:
```bash
# 1. Create and activate a virtual environment
python3 -m venv samatnext-env
source samatnext-env/bin/activate

# 2. Install MLX and scientific dependencies
pip install mlx datasets tokenizers
```

### B. Run the Pipeline
We provide simple scripts to run the full training and evaluation steps:

```bash
# 1. Pretrain both models on codeparrot subset (1,000 steps, ~3.75 hours)
PYTHONPATH=. ./samatnext-env/bin/python scripts/pretrain.py

# 2. Fine-tune both models on python-codes-25k (500 steps)
PYTHONPATH=. ./samatnext-env/bin/python scripts/finetune.py

# 3. Run the token-level code completion benchmark
PYTHONPATH=. ./samatnext-env/bin/python scripts/benchmark.py

# 4. Run the 80-task execution-based functional correctness (pass@1) benchmark
PYTHONPATH=. ./samatnext-env/bin/python scripts/execution_benchmark.py
```

*Evaluation results are automatically printed to the stdout and saved in JSON format under `results/execution_benchmark_results.json`.*

---

## 5. Project Layout

```
├── model/
│   ├── config_120m.py          # 120M experiment configs
│   ├── config.py               # 520M locked configurations
│   ├── baseline_model.py       # Baseline Transformer model
│   ├── samatnext_model.py      # SamatNext hybrid architecture
│   ├── diff_attn_layer.py      # Differential Attention module
│   ├── delta_layer.py          # Gated Causal Attention module
│   ├── memory_bus.py           # Cross-Layer Memory Bus module
│   ├── mtp_head.py             # Multi-Token Prediction module
│   ├── latent_tokens.py        # Learned Soft Prompts module
│   └── mlp.py                  # SwiGLU MLP implementation
├── train/
│   └── experiment_trainer.py   # Training engine with gradient accumulation
├── data/
│   ├── dataset.py              # Dataset parser and iterator
│   ├── tokenizer.py            # Custom BPE wrapper
│   └── fim.py                  # Fill-in-the-Middle transformer
├── scripts/
│   ├── pretrain.py             # Pretraining script
│   ├── finetune.py             # Fine-tuning script
│   ├── benchmark.py            # Token continuation evaluator
│   └── execution_benchmark.py  # 80-problem pass@1 execution harness
├── ARCHITECTURE.md             # Formal LaTeX mathematical specifications
├── SPEC.md                     # Locked parameter specifications
└── README.md                   # This file
```

---

## 6. License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
