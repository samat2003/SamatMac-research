"""Shared evaluation harness for the 120M Validation Experiment."""

import mlx.core as mx
import mlx.nn as nn
from data.tokenizer import SamatNextTokenizer
from model.config_120m import Experiment120MConfig

PROMPTS = [
    "def fibonacci(n):",
    "class User:\n    def __init__(self, name):",
    "# A python function to sum all even numbers in a list\ndef sum_evens(nums):",
]

FIM_PROMPTS = [
    {"prefix": "def get_api_url():\n    return '", "suffix": "'\n", "middle_hint": "https://api.example.com"},
]


def evaluate_loss(model: nn.Module, dataset, num_batches: int = 10, batch_size: int = 4) -> float:
    """Evaluate held-out loss/perplexity on a subset of the dataset."""
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(dataset.iterate_batches(batch_size)):
        if i >= num_batches:
            break
        
        input_ids = mx.array(batch.input_ids, dtype=mx.int32)
        targets = mx.array(batch.targets, dtype=mx.int32)
        loss_mask = mx.array(batch.loss_mask, dtype=mx.bfloat16)
        
        out = model(input_ids, targets=targets, loss_mask=loss_mask)
        mx.eval(out["lm_loss"])
        total_loss += out["lm_loss"].item()
        count += 1
        
    if count == 0:
        return float('inf')
    return total_loss / count


def generate_greedy(model: nn.Module, tokenizer: SamatNextTokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Simple greedy generation without KV cache (as specified in architecture docs)."""
    # model.latent_tokens state is reset at the start of __call__ due to self.memory_bus.reset()
    input_ids = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
    
    generated = []
    for _ in range(max_new_tokens):
        out = model(input_ids)
        logits = out["logits"]
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]], dtype=mx.int32)], axis=1)
        
        if next_token == tokenizer.tokenizer.token_to_id("<eos>") or next_token == tokenizer.tokenizer.token_to_id("<|endoftext|>"):
            break
            
    return tokenizer.decode(generated)


def evaluate_generations(model: nn.Module, tokenizer: SamatNextTokenizer) -> dict:
    """Evaluate small set of prompts and return generated text."""
    results = {}
    
    # 1. Code completion
    for i, prompt in enumerate(PROMPTS):
        gen = generate_greedy(model, tokenizer, prompt, max_new_tokens=32)
        results[f"completion_{i}"] = {"prompt": prompt, "generated": gen}
        
    # 2. FIM
    fim_prefix = tokenizer.tokenizer.token_to_id("<fim_prefix>")
    fim_suffix = tokenizer.tokenizer.token_to_id("<fim_suffix>")
    fim_middle = tokenizer.tokenizer.token_to_id("<fim_middle>")
    
    if all(x is not None for x in [fim_prefix, fim_suffix, fim_middle]):
        for i, f_prompt in enumerate(FIM_PROMPTS):
            # Format: <fim_prefix> prefix <fim_suffix> suffix <fim_middle>
            full_prompt = f"<fim_prefix>{f_prompt['prefix']}<fim_suffix>{f_prompt['suffix']}<fim_middle>"
            gen = generate_greedy(model, tokenizer, full_prompt, max_new_tokens=16)
            results[f"fim_{i}"] = {"prompt": full_prompt, "generated": gen}
            
    return results


def run_full_eval(model: nn.Module, tokenizer: SamatNextTokenizer, val_dataset) -> dict:
    loss = evaluate_loss(model, val_dataset, num_batches=10)
    perplexity = mx.exp(loss).item() if loss < 100 else float('inf')
    
    generations = evaluate_generations(model, tokenizer)
    
    return {
        "val_loss": loss,
        "val_perplexity": perplexity,
        "generations": generations
    }
