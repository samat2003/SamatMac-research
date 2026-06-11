"""Simple Code Completion Benchmark for small models.

Tests whether the model can complete partial Python code,
rather than follow natural language instructions.
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

import mlx.core as mx

from data.tokenizer import SamatNextTokenizer
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import BaselineModel
from model.samatnext_model import SamatNextModel

# Simple code completion tasks — give a prefix, check if the completion is reasonable
TASKS = [
    # 1. Can it complete a return statement?
    {
        "name": "Return sum",
        "prompt": "def add(a, b):\n    return a",
        "accept": ["+", " +", " + b"],
        "mode": "starts_with",
    },
    # 2. Can it complete a print statement?
    {
        "name": "Print hello",
        "prompt": "print('hello",
        "accept": ["')", " world')", "')"],
        "mode": "starts_with",
    },
    # 3. Does it know what comes after 'for i in'?
    {
        "name": "For loop",
        "prompt": "for i in range(",
        "accept": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "n", "len"],
        "mode": "starts_with",
    },
    # 4. Can it complete an if statement?
    {
        "name": "If condition",
        "prompt": "if x ==",
        "accept": [" 0", " 1", " True", " False", " None", " \"", " '"],
        "mode": "starts_with",
    },
    # 5. Does it produce valid Python after def?
    {
        "name": "Function def",
        "prompt": "def multiply(x, y):\n    return x",
        "accept": [" *", " * y", "*y", " * ", "*"],
        "mode": "starts_with",
    },
    # 6. List comprehension
    {
        "name": "List comprehension",
        "prompt": "squares = [x**2 for x in",
        "accept": [" range", " list", " [", " nums", " data", " arr"],
        "mode": "starts_with",
    },
    # 7. Import completion
    {
        "name": "Import os",
        "prompt": "import os\npath = os.path.",
        "accept": ["join", "exists", "dirname", "basename", "abspath", "isfile", "isdir"],
        "mode": "starts_with",
    },
    # 8. String method
    {
        "name": "String split",
        "prompt": "words = sentence.split(",
        "accept": [")", "'", "\" \"", "' '", "\",\""],
        "mode": "starts_with",
    },
    # 9. Dictionary access
    {
        "name": "Dict access",
        "prompt": "value = my_dict[",
        "accept": ["'", "\"", "key", "'key'", "\"key\""],
        "mode": "starts_with",
    },
    # 10. Simple arithmetic assignment
    {
        "name": "Arithmetic",
        "prompt": "x = 10\ny = 20\nresult = x +",
        "accept": [" y", " 1", " 2", " 10", " 20"],
        "mode": "starts_with",
    },
    # 11. Class definition
    {
        "name": "Class init",
        "prompt": "class Dog:\n    def __init__(self",
        "accept": [",", ", name", ")"],
        "mode": "starts_with",
    },
    # 12. Try/except
    {
        "name": "Try except",
        "prompt": "try:\n    result = int(x)\nexcept",
        "accept": [" Value", " Exception", " Type", ":"],
        "mode": "starts_with",
    },
    # 13. While loop
    {
        "name": "While loop",
        "prompt": "while i <",
        "accept": [" n", " 10", " len", " max", " 100", " N"],
        "mode": "starts_with",
    },
    # 14. Return boolean
    {
        "name": "Return bool",
        "prompt": "def is_empty(lst):\n    return len(lst) ==",
        "accept": [" 0", "0"],
        "mode": "starts_with",
    },
    # 15. F-string
    {
        "name": "F-string",
        "prompt": "name = 'Alice'\nprint(f'Hello, {",
        "accept": ["name", "name}"],
        "mode": "starts_with",
    },
]


def generate_tokens(model, tokenizer, prompt, max_tokens=20):
    """Generate a short completion and return it as a string."""
    prompt_ids = tokenizer.encode(prompt)
    x = mx.array(prompt_ids, dtype=mx.int32)[None, :]

    generated_ids = []
    for _ in range(max_tokens):
        out = model(x)
        logits = out["logits"][:, -1, :]
        next_token = mx.argmax(logits, axis=-1).item()

        if next_token == tokenizer.eos_id:
            break

        generated_ids.append(next_token)
        x = mx.concatenate([x, mx.array([[next_token]], dtype=mx.int32)], axis=1)

    return tokenizer.decode(generated_ids)


def run_benchmark(model_name: str, seed: int = 42):
    print(f"\n{'='*60}\nBenchmarking {model_name} (Code Completion)\n{'='*60}")

    if model_name == "Baseline-120M":
        config = Baseline120MConfig()
        model = BaselineModel(config)
    else:
        config = SamatNext120MConfig()
        model = SamatNextModel(config)

    tokenizer = SamatNextTokenizer.from_file("data/tokenizer.json", config=config)

    # Try fine-tuned weights first, fall back to pre-trained
    finetune_ckpt = Path(f"results/finetune_120m/{model_name}_seed_{seed}/step_000500.npz")
    pretrain_ckpt = Path(f"results/pretrain_120m/{model_name}_seed_{seed}/step_001000.npz")

    if finetune_ckpt.exists():
        print(f"Loading fine-tuned weights: {finetune_ckpt}")
        model.load_weights(str(finetune_ckpt))
    elif pretrain_ckpt.exists():
        print(f"Loading pre-trained weights: {pretrain_ckpt}")
        model.load_weights(str(pretrain_ckpt))
    else:
        print("WARNING: No checkpoint found, using random weights!")

    passed = 0
    total = len(TASKS)

    for i, task in enumerate(TASKS):
        completion = generate_tokens(model, tokenizer, task["prompt"], max_tokens=20)
        # Clean: take only the first line of completion
        first_line = completion.split("\n")[0]

        matched = False
        for accept in task["accept"]:
            if first_line.startswith(accept):
                matched = True
                break

        status = "✅ PASS" if matched else "❌ FAIL"
        if matched:
            passed += 1

        print(f"  {i+1:2d}. {task['name']:20s} | {status} | Completed: {repr(first_line[:60])}")

    score = (passed / total) * 100
    print(f"\n  Score for {model_name}: {passed}/{total} ({score:.1f}%)")
    return passed, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    b_pass, b_total = run_benchmark("Baseline-120M", args.seed)
    s_pass, s_total = run_benchmark("SamatNext-120M", args.seed)

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline-120M:   {b_pass}/{b_total} ({b_pass/b_total*100:.1f}%)")
    print(f"  SamatNext-120M:  {s_pass}/{s_total} ({s_pass/s_total*100:.1f}%)")
    print(f"{'='*60}")
