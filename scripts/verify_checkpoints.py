"""Verify both checkpoints load correctly and compare generation quality.

Loads pretrain and finetune checkpoints for both Baseline-120M and SamatNext-120M,
validates parameter integrity, and generates Python completions from a battery of prompts.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlx.core as mx
from mlx.utils import tree_flatten

from data.tokenizer import SamatNextTokenizer
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import BaselineModel
from model.samatnext_model import SamatNextModel


# ── Prompts for coherence testing ──────────────────────────────────────────────
PROMPTS = [
    # Basic code completion
    "def fibonacci(n):\n    ",
    # Function with logic
    "def is_palindrome(s):\n    return s ==",
    # Class definition
    "class Stack:\n    def __init__(self):\n        self.",
    # Import + usage
    "import os\nfiles = os.listdir(",
    # Multi-line function
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return",
    # List comprehension
    "numbers = [1, 2, 3, 4, 5]\neven = [x for x in numbers if",
    # Error handling
    "try:\n    value = int(input())\nexcept ValueError",
    # Decorator pattern
    "def timer(func):\n    def wrapper(*args):\n        start = time.",
]

SEPARATOR = "=" * 70


def count_params(model):
    """Count total trainable parameters."""
    total = 0
    for k, v in tree_flatten(model.parameters()):
        total += v.size
    return total


def check_weight_health(model, model_name):
    """Check for NaN/Inf values in model weights."""
    issues = []
    total_params = 0
    for k, v in tree_flatten(model.parameters()):
        total_params += v.size
        v_eval = mx.eval(v)
        if mx.any(mx.isnan(v)).item():
            issues.append(f"  ⚠ NaN found in {k}")
        if mx.any(mx.isinf(v)).item():
            issues.append(f"  ⚠ Inf found in {k}")
    
    if issues:
        print(f"  ❌ {model_name}: {len(issues)} weight health issue(s):")
        for issue in issues:
            print(issue)
    else:
        print(f"  ✅ {model_name}: All weights healthy ({total_params:,} params)")
    return len(issues) == 0


def generate(model, tokenizer, prompt, max_new_tokens=64):
    """Greedy generation."""
    input_ids = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
    
    generated = []
    for _ in range(max_new_tokens):
        out = model(input_ids)
        logits = out["logits"]
        next_token = mx.argmax(logits[0, -1, :]).item()
        
        if next_token == tokenizer.eos_id:
            break
        
        generated.append(next_token)
        input_ids = mx.concatenate(
            [input_ids, mx.array([[next_token]], dtype=mx.int32)], axis=1
        )
    
    return tokenizer.decode(generated)


def load_model_and_weights(model_name, ckpt_path, config, ModelClass):
    """Instantiate model and load checkpoint weights."""
    model = ModelClass(config)
    
    print(f"\n  Loading: {ckpt_path}")
    t0 = time.time()
    model.load_weights(str(ckpt_path))
    mx.eval(model.parameters())
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    
    return model


def run_generation_comparison(models, tokenizer, prompts):
    """Run generation on all prompts for all models and collect results."""
    results = {}
    
    for label, model in models.items():
        results[label] = []
        for prompt in prompts:
            t0 = time.time()
            completion = generate(model, tokenizer, prompt, max_new_tokens=64)
            elapsed = time.time() - t0
            results[label].append({
                "prompt": prompt,
                "completion": completion,
                "time": elapsed,
            })
    
    return results


def print_comparison(results, prompts):
    """Print side-by-side generation comparison."""
    labels = list(results.keys())
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'─' * 70}")
        print(f"PROMPT {i+1}:")
        print(f"  {repr(prompt[:80])}")
        print()
        
        for label in labels:
            entry = results[label][i]
            comp = entry["completion"]
            # Show first 3 lines of completion
            lines = comp.split("\n")[:4]
            display = "\n    ".join(lines)
            print(f"  [{label}] ({entry['time']:.2f}s):")
            print(f"    {display}")
            print()


def score_coherence(completion):
    """Basic heuristic coherence score (0-5)."""
    score = 0
    
    # 1. Non-empty
    if completion.strip():
        score += 1
    
    # 2. Contains Python keywords/syntax
    py_keywords = ["def ", "return ", "if ", "for ", "class ", "import ", "self.", "print(", "=", "(", ")"]
    if any(kw in completion for kw in py_keywords):
        score += 1
    
    # 3. No excessive repetition (sign of degenerate output)
    words = completion.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.3:
            score += 1
    elif len(words) > 0:
        score += 1  # too short to judge repetition
    
    # 4. Syntactically reasonable (try compiling)
    # We wrap in a function to avoid import/runtime errors
    try:
        compile(completion, "<string>", "exec")
        score += 1
    except SyntaxError:
        # Partial code may not compile — try as a continuation
        pass
    
    # 5. Length is reasonable (not empty, not endlessly long)
    if 5 < len(completion) < 500:
        score += 1
    
    return score


def main():
    print(SEPARATOR)
    print("CHECKPOINT VERIFICATION & MODEL COMPARISON")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEPARATOR)
    
    # ── 1. Setup configs ───────────────────────────────────────────────────
    baseline_cfg = Baseline120MConfig()
    samatnext_cfg = SamatNext120MConfig()
    
    print(f"\n[Config] Baseline-120M: d_model={baseline_cfg.d_model}, "
          f"n_layers={baseline_cfg.n_layers}, d_ffn={baseline_cfg.d_ffn}, "
          f"max_seq_len={baseline_cfg.max_seq_len}")
    print(f"[Config] SamatNext-120M: d_model={samatnext_cfg.d_model}, "
          f"n_layers={samatnext_cfg.n_layers}, d_ffn={samatnext_cfg.d_ffn}, "
          f"mtp_heads={samatnext_cfg.mtp_heads}, max_seq_len={samatnext_cfg.max_seq_len}")
    
    # ── 2. Load tokenizer ──────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("LOADING TOKENIZER")
    print(SEPARATOR)
    tokenizer = SamatNextTokenizer.from_file("data/tokenizer.json", config=baseline_cfg)
    print(f"  Tokenizer vocab size: {tokenizer.tokenizer.get_vocab_size()}")
    print(f"  BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}, PAD={tokenizer.pad_id}")
    
    # Quick encode/decode sanity
    test_str = "def hello():\n    print('world')"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded[1:])  # skip BOS
    print(f"  Encode/decode test: {len(encoded)} tokens → '{decoded[:50]}'")
    
    # ── 3. Load all checkpoints ────────────────────────────────────────────
    checkpoints = {
        "Baseline-120M (pretrain)": {
            "path": "results/pretrain_120m/Baseline-120M_seed_42/step_001000.npz",
            "config": baseline_cfg,
            "cls": BaselineModel,
        },
        "Baseline-120M (finetune)": {
            "path": "results/finetune_120m/Baseline-120M_seed_42/step_000500.npz",
            "config": baseline_cfg,
            "cls": BaselineModel,
        },
        "SamatNext-120M (pretrain)": {
            "path": "results/pretrain_120m/SamatNext-120M_seed_42/step_001000.npz",
            "config": samatnext_cfg,
            "cls": SamatNextModel,
        },
        "SamatNext-120M (finetune)": {
            "path": "results/finetune_120m/SamatNext-120M_seed_42/step_000500.npz",
            "config": samatnext_cfg,
            "cls": SamatNextModel,
        },
    }
    
    models = {}
    all_healthy = True
    
    for label, info in checkpoints.items():
        print(f"\n{SEPARATOR}")
        print(f"LOADING: {label}")
        print(SEPARATOR)
        
        if not os.path.exists(info["path"]):
            print(f"  ❌ MISSING: {info['path']}")
            continue
        
        size_mb = os.path.getsize(info["path"]) / (1024 * 1024)
        print(f"  Checkpoint size: {size_mb:.1f} MB")
        
        try:
            model = load_model_and_weights(label, info["path"], info["config"], info["cls"])
            n_params = count_params(model)
            print(f"  Parameter count: {n_params:,}")
            
            healthy = check_weight_health(model, label)
            all_healthy = all_healthy and healthy
            
            models[label] = model
        except Exception as e:
            print(f"  ❌ FAILED to load: {e}")
            import traceback
            traceback.print_exc()
    
    # ── 4. Parameter count comparison ──────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("PARAMETER COUNT SUMMARY")
    print(SEPARATOR)
    for label, model in models.items():
        n = count_params(model)
        print(f"  {label:35s}  {n:>12,}")
    
    # ── 5. Generation comparison (finetune checkpoints preferred) ──────────
    # Use finetuned models for generation comparison
    gen_models = {}
    for label in ["Baseline-120M (finetune)", "SamatNext-120M (finetune)"]:
        if label in models:
            gen_models[label] = models[label]
    
    # Also test pretrain models if finetune didn't load
    for label in ["Baseline-120M (pretrain)", "SamatNext-120M (pretrain)"]:
        short = label.replace(" (pretrain)", "")
        finetune_label = f"{short} (finetune)"
        if finetune_label not in gen_models and label in models:
            gen_models[label] = models[label]
    
    if gen_models:
        print(f"\n{SEPARATOR}")
        print("GENERATION COMPARISON (Finetuned Models)")
        print(SEPARATOR)
        
        results = run_generation_comparison(gen_models, tokenizer, PROMPTS)
        print_comparison(results, PROMPTS)
        
        # ── 6. Coherence scoring ───────────────────────────────────────────
        print(f"\n{SEPARATOR}")
        print("COHERENCE SCORES (0-5 heuristic)")
        print(SEPARATOR)
        
        for label in results:
            scores = []
            for entry in results[label]:
                s = score_coherence(entry["completion"])
                scores.append(s)
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {label:35s}  avg={avg:.2f}/5  scores={scores}")
    
    # ── 7. Also compare pretrain vs finetune for each arch ─────────────────
    for arch_name, pretrain_label, finetune_label in [
        ("Baseline-120M", "Baseline-120M (pretrain)", "Baseline-120M (finetune)"),
        ("SamatNext-120M", "SamatNext-120M (pretrain)", "SamatNext-120M (finetune)"),
    ]:
        if pretrain_label in models and finetune_label in models:
            print(f"\n{SEPARATOR}")
            print(f"PRETRAIN vs FINETUNE: {arch_name}")
            print(SEPARATOR)
            
            pair = {pretrain_label: models[pretrain_label], finetune_label: models[finetune_label]}
            pair_results = run_generation_comparison(pair, tokenizer, PROMPTS[:3])
            print_comparison(pair_results, PROMPTS[:3])
    
    # ── 8. Run the code completion benchmark ───────────────────────────────
    print(f"\n{SEPARATOR}")
    print("CODE COMPLETION BENCHMARK (from benchmark.py tasks)")
    print(SEPARATOR)
    
    from scripts.benchmark import TASKS
    
    summary = {}
    for label, model in gen_models.items():
        passed = 0
        for task in TASKS:
            prompt_ids = tokenizer.encode(task["prompt"])
            x = mx.array(prompt_ids, dtype=mx.int32)[None, :]
            
            generated_ids = []
            for _ in range(20):
                out = model(x)
                logits = out["logits"][:, -1, :]
                next_token = mx.argmax(logits, axis=-1).item()
                if next_token == tokenizer.eos_id:
                    break
                generated_ids.append(next_token)
                x = mx.concatenate([x, mx.array([[next_token]], dtype=mx.int32)], axis=1)
            
            completion = tokenizer.decode(generated_ids)
            first_line = completion.split("\n")[0]
            
            matched = any(first_line.startswith(a) for a in task["accept"])
            status = "✅" if matched else "❌"
            if matched:
                passed += 1
            print(f"  [{label[:20]:20s}] {task['name']:20s} {status}  → {repr(first_line[:50])}")
        
        score = (passed / len(TASKS)) * 100
        summary[label] = {"passed": passed, "total": len(TASKS), "score": score}
        print(f"  → {label}: {passed}/{len(TASKS)} ({score:.1f}%)\n")
    
    # ── 9. Final summary ──────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("FINAL SUMMARY")
    print(f"{'━' * 70}")
    print(f"  Checkpoints loaded: {len(models)}/4")
    print(f"  Weight health: {'✅ All healthy' if all_healthy else '⚠ Issues found'}")
    
    if summary:
        print(f"\n  Benchmark Results:")
        for label, s in summary.items():
            print(f"    {label:35s}  {s['passed']}/{s['total']} ({s['score']:.1f}%)")
    
    print(f"{'━' * 70}")


if __name__ == "__main__":
    main()
