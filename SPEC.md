# SamatNext-120M - Project Specification

## Hardware Target
- Device: Apple iMac, Apple Silicon M4
- Unified memory: 16GB
- Training backend: MLX (primary)
- Inference target: Mac-native local generation

## Software Environment
- Python: 3.11+
- MLX: 0.18.0 or higher
- Tokenizers: 0.19.0 or higher
- Datasets: 2.20.0 or higher
- human-eval: 1.0.3 or higher

## Model Identity
- Name: SamatNext-120M & Baseline-120M
- Architecture: Hybrid Gated Causal Attention + Differential Attention (SamatNext) vs Standard Transformer (Baseline)
- Parameter count: 116,883,461 (SamatNext) and 116,583,168 (Baseline)
- Sequence complexity: quadratic; both attention mixers materialize full causal softmax maps
- Runtime activations: bfloat16; parameter/checkpoint storage: float32
- No dropout at any stage

## Architecture Constants (120M Spec)
- d_model: 768
- n_layers: 10 (SamatNext alternates 5 gated causal and 5 differential layers; Baseline uses standard attention layers)
- n_heads: 12
- n_kv_heads: 4 (Grouped-Query Attention, GQA)
- d_ffn (SamatNext-120M): 2700 (SwiGLU, actual gated dimension)
- d_ffn (Baseline-120M): 3310 (wider to match parameter counts)
- vocab_size: 32000
- max_seq_len: 512
- delta_chunk_size: 64 (reserved)
- delta_sparse_gate: True
- memory_bus_layers: 2 (layers 8 and 10 write and read causal prefix states)
- diff_attn_lambda_init: 0.8
- rope_base: 10000.0
- yarn_scale: 1.0
- yarn_alpha: 1.0
- n_latent_tokens: 4
- mtp_heads: 2
- mtp_confidence_gate: True
- tie_embeddings: True (vocabulary embedding and output projection share weights)

## Training Constants
- batch_size: 4
- gradient_accumulation_steps: 4 (pretrain / finetune)
- effective_batch_size: 16
- learning_rate: 3e-4
- lr_schedule: cosine with warmup
- warmup_steps: 500
- max_steps: 1000 (pretrain) and 500 (finetune)
- optimizer: AdamW
- adam_beta1: 0.9
- adam_beta2: 0.95
- adam_epsilon: 1e-8
- weight_decay: 0.1
- grad_clip: 1.0

## Data Constants
- Language: Python only
- FIM rate: 0.50 (50% of samples use fill-in-the-middle)
- FIM tokens: <fim_prefix>, <fim_suffix>, <fim_middle>
- Pretraining dataset: GitHub Clean Code (Python subset) - 100,000 samples (~8M tokens)
- Fine-tuning dataset: Python-Codes-25k (~50K instruction+code pairs)
- Deduplication: exact match on first 50 chars + length bucket

## Memory Budget (16GB unified)
- Serialized model weights FP32: ~445MB (.npz)
- AdamW buffers and moment storage: ~1.3GB
- Parameters + moments + gradients: ~1.8GB
- Attention activations: quadratic in sequence length
- No KV cache is currently implemented for evaluation/generation (recomputes sequence)
- OS overhead: ~3-4GB
- Batch 4 x 512 training runs safely within 16GB unified memory

## Evaluation Metrics
- Primary: Token-Level Code Completion (15 tasks)
- Secondary: 80-task Execution-Based Functional Correctness (pass@1 with greedy decoding)

## Project Structure
```
samatnext/
├── SPEC.md
├── ARCHITECTURE.md
├── README.md
├── model/
│   ├── config_120m.py
│   ├── config.py
│   ├── delta_layer.py
│   ├── diff_attn_layer.py
│   ├── mlp.py
│   ├── memory_bus.py
│   ├── latent_tokens.py
│   ├── mtp_head.py
│   ├── baseline_model.py
│   ├── samatnext_model.py
├── data/
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── fim.py
│   ├── python_codes_25k.jsonl
├── scripts/
│   ├── pretrain.py
│   ├── finetune.py
│   ├── benchmark.py
│   ├── execution_benchmark.py
│   ├── verify_checkpoints.py
│   ├── test_step_speed.py
```

## Critical Rules
- Describe the even-layer mixer accurately as quadratic gated causal softmax attention
- Never import torch in any file inside model/ (MLX only)
- Use mx.bfloat16 for runtime activations; FP32 parameter storage
- Never hardcode any constant that exists in SPEC.md. Always import from model/config_120m.py (or model/config.py for 520M target)
- No dropout layers anywhere
- No learned positional embeddings (RoPE only)
