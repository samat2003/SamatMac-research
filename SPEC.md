# SamatNext-520M - Project Specification

## Hardware Target
- Device: Apple iMac, Apple Silicon M4
- Unified memory: 16GB
- Training backend: MLX (primary), PyTorch MPS (fallback/eval only)
- Inference target: Mac-native, no internet required

## Software Environment
- Python: 3.11
- MLX: 0.18.0 or higher
- PyTorch: 2.3.0 or higher (MPS backend)
- Transformers: 4.44.0 or higher
- Tokenizers: 0.19.0 or higher
- Datasets: 2.20.0 or higher
- SentencePiece: 0.2.0 or higher
- NumPy: 1.26.0 or higher
- WandB: 0.17.0 or higher
- HuggingFace Hub: 0.24.0 or higher
- Evaluate: 0.4.0 or higher
- Accelerate: 0.31.0 or higher
- human-eval: 1.0.3 or higher
- Rich: 13.7.0 or higher
- SlidingSpeed: local install from github.com/samat2003/SlidingSpeed

## Model Identity
- Name: SamatNext-520M
- Architecture: Hybrid Gated Causal Attention + Differential Attention
- Parameter count: 520,930,316 serialized trainable elements
- Sequence complexity: quadratic; both mixers materialize causal attention maps
- Runtime activations: bfloat16; current parameters/checkpoints: float32
- No dropout at any stage

## Architecture Constants (locked, do not change)
- d_model: 1024
- n_layers: 24 (even index = gated causal attention, odd index = DiffAttn)
- n_heads: 16
- n_kv_heads: 4 (GQA, applies to DiffAttn layers only)
- d_ffn: 4096 (SwiGLU, actual gated dim)
- vocab_size: 32000
- max_seq_len: 4096
- delta_chunk_size: 64 (reserved; not used by the current attention mixer)
- delta_sparse_gate: True (implemented as a dense sigmoid output gate)
- memory_bus_layers: 4 (last 4 gated-attention layers write causal prefix states)
- diff_attn_lambda_init: 0.8
- rope_base: 10000.0
- yarn_scale: 1.0
- yarn_alpha: 1.0
- n_latent_tokens: 4
- mtp_heads: 6
- mtp_confidence_gate: True
- tie_embeddings: False (the three vocabulary matrices train independently)

## Training Constants (locked)
- batch_size: 4
- gradient_accumulation_steps: 8
- effective_batch_size: 32
- learning_rate: 3e-4
- lr_schedule: cosine with warmup
- warmup_steps: 500
- max_steps: 100000
- optimizer: AdamW
- adam_beta1: 0.9
- adam_beta2: 0.95
- adam_epsilon: 1e-8
- weight_decay: 0.1
- grad_clip: 1.0
- save_every_steps: 1000
- eval_every_steps: 500

## Data Constants (locked)
- Language: Python only
- FIM rate: 0.50 (50% of samples use fill-in-the-middle)
- FIM tokens: <fim_prefix>, <fim_suffix>, <fim_middle>
- Distillation rate: 30% of training data from GPT-4o mini generated samples
- Max tokens per sample: 4096
- Dataset sources: The Stack v2 Python filtered, CodeParrot Python clean, GPT-4o mini synthetic
- Deduplication: required, exact match on first 50 chars + length bucket

## Memory Budget (16GB unified)
- Serialized model weights FP32: ~2.08GB
- AdamW parameter and moment storage: approximately 6.25GB before gradients
- Parameters + AdamW moments + gradients: approximately 8.33GB
- Attention activations: workload-dependent and quadratic in sequence length
- No KV cache is currently implemented for evaluation/generation
- OS overhead: ~3-4GB
- Full batch 4 x 4096 training is not yet validated within 16GB unified memory
- Pretraining must begin with measured sequence-length and batch-size scaling

## Evaluation Targets
- Primary: HumanEval pass@1
- Secondary: MBPP pass@1
- Tertiary: HumanEval+ pass@1
- Target score: 82-88% HumanEval pass@1

## Project Structure
samatnext/
├── SPEC.md
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── delta_layer.py
│   ├── diff_attn_layer.py
│   ├── mlp.py
│   ├── memory_bus.py
│   ├── latent_tokens.py
│   ├── mtp_head.py
│   ├── model.py
├── data/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── fim.py
│   ├── dataset.py
│   ├── synthetic.py
├── train/
│   ├── __init__.py
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── sliding_mac.py
├── eval/
│   ├── __init__.py
│   ├── humaneval.py
│   ├── mbpp.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── eval.py
│   ├── generate.py

## Critical Rules (agent must follow these in every file)
- Describe the even-layer mixer accurately as quadratic gated causal softmax attention
- Never import torch in any file inside model/ (MLX only)
- Use mx.bfloat16 for runtime activations; document FP32 parameter storage
- Never hardcode any constant that exists in SPEC.md. Always import from model/config.py
- Every file must have a module docstring referencing SPEC.md
- No dropout layers anywhere
- No learned positional embeddings. RoPE + YaRN only
