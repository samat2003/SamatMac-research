"""HumanEval evaluation harness for SamatNext-520M per SPEC.md."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from data.tokenizer import SamatNextTokenizer
from model.config import DEFAULT_CONFIG, SamatNextConfig
from model.model import SamatNext


MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95


def sample_token(logits: mx.array, temperature: float, top_p: float) -> int:
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    probs_np = np.array(probs)

    sorted_indices = np.argsort(probs_np)[::-1]
    sorted_probs = probs_np[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, top_p, side="right") + 1
    cutoff = min(cutoff, len(sorted_indices))

    filtered_probs = np.zeros_like(probs_np)
    kept_indices = sorted_indices[:cutoff]
    filtered_probs[kept_indices] = probs_np[kept_indices]
    filtered_probs = filtered_probs / filtered_probs.sum()

    token_id = np.random.choice(len(filtered_probs), p=filtered_probs)
    return int(token_id)


def generate(
    model: SamatNext,
    tokenizer: SamatNextTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    input_ids = tokenizer.encode(prompt, add_bos=True)
    original_len = len(input_ids)

    for _ in range(max_new_tokens):
        model_input = mx.array([input_ids], dtype=mx.int32)
        out = model(model_input)
        next_logits = out["logits"][0, -1, :]
        mx.eval(next_logits)
        next_token = sample_token(next_logits, temperature, top_p)
        if next_token == tokenizer.eos_id:
            break
        input_ids.append(next_token)

    return tokenizer.decode(input_ids[original_len:])


def evaluate_humaneval(
    model: SamatNext,
    tokenizer: SamatNextTokenizer,
    problems_path: str = None,
    n_problems: int = None,
) -> dict:
    if problems_path is not None:
        problems = {}
        with open(problems_path, "r", encoding="utf-8") as file:
            for line in file:
                problem = json.loads(line)
                problems[problem["task_id"]] = problem
    else:
        from human_eval.data import read_problems

        problems = read_problems()

    problem_items = list(problems.items())
    if n_problems is not None:
        problem_items = problem_items[:n_problems]

    completions = []
    for index, (task_id, problem) in enumerate(problem_items, start=1):
        prompt = problem["prompt"]
        completion = generate(model, tokenizer, prompt)
        completions.append(
            {
                "task_id": task_id,
                "completion": completion,
            }
        )
        if index % 10 == 0:
            print(f"Generated {index}/{len(problem_items)} completions")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as file:
        temp_file = file.name
        for completion in completions:
            file.write(json.dumps(completion) + "\n")

    from human_eval.evaluation import evaluate_functional_correctness

    results = evaluate_functional_correctness(temp_file)
    return {
        "pass@1": results["pass@1"],
        "n_problems": len(completions),
        "completions_file": temp_file,
    }


def check_eval(config: SamatNextConfig) -> None:
    print("HumanEval harness: checking imports...")
    try:
        from human_eval.data import read_problems

        print("human_eval installed OK")
    except ImportError:
        print("human_eval not installed. Run: pip install human-eval")
    print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print(f"TOP_P: {TOP_P}")
    print("HumanEval OK")


if __name__ == "__main__":
    check_eval(DEFAULT_CONFIG)
