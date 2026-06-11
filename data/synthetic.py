"""Synthetic distillation pipeline for SamatNext-520M per SPEC.md."""

import json
import random
import time
from pathlib import Path
from typing import Optional

import requests

from model.config import DEFAULT_CONFIG, SamatNextConfig


OLLAMA_API_URL = "http://localhost:11434/api/chat"
TEACHER_MODEL = "qwen2.5-coder:7b-instruct"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

SYSTEM_PROMPT = """You are an expert Python programmer and teacher. When given a programming task:
1. Write a complete, correct Python solution
2. Add a brief reasoning comment before each non-trivial line explaining WHY not just what
3. Choose optimal data structures and explain the choice
4. End with a one-line docstring summary
Format: only return raw Python code with inline comments. No markdown. No explanations outside the code."""

PROBLEM_TEMPLATES = [
    "Write a function that reverses the order of words in a sentence while preserving each word's characters and collapsing repeated whitespace.",
    "Write a function that determines whether a string is a palindrome while ignoring punctuation, whitespace, and letter case.",
    "Write a function that groups a list of words into anagram groups and returns each group sorted alphabetically.",
    "Write a stable function that sorts a list of employee dictionaries by descending salary and then ascending last name.",
    "Write a function that filters invalid integers from a mixed list and returns only unique even integers in their original order.",
    "Write a function that removes duplicate dictionaries from a list using the values of the 'id' key while preserving the first occurrence.",
    "Write a function that inverts a dictionary whose values may repeat, mapping each value to a sorted list of its original keys.",
    "Write a function that recursively merges two nested dictionaries, adding numeric conflicts and preferring the second value for other conflicts.",
    "Write a function that groups transaction dictionaries by customer_id and returns the total amount for each customer.",
    "Write a memoized recursive function that returns the nth Fibonacci number and rejects negative inputs with ValueError.",
    "Write a recursive factorial function with type hints that validates the input is a non-negative integer.",
    "Define a binary tree node dataclass and write a recursive in-order traversal that returns the node values as a list.",
    "Write an iterative binary search function that returns the first index of a target in a sorted list containing duplicates.",
    "Write a two-pointer function that returns all unique pairs of integers whose sum equals a target, without modifying the input list.",
    "Write a sliding-window function that finds the length and substring of the longest section containing at most two distinct characters.",
    "Write a function that reads a CSV file of sales records, validates required columns, and returns total revenue grouped by product.",
    "Write a function that loads a JSON file, reports a clear error for malformed JSON, and validates that the top-level value is a dictionary.",
    "Implement a Stack class with push, pop, peek, length, and empty checks, raising IndexError for invalid pop or peek operations.",
    "Implement a linked-list-backed FIFO Queue class with O(1) enqueue and dequeue operations using a private node dataclass.",
    "Write a lazy generator that uses itertools to yield fixed-size batches from any iterable, including a final smaller batch.",
]


class SyntheticDataGenerator:
    def __init__(
        self,
        config: SamatNextConfig,
        output_dir: str = "data/synthetic_cache",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(42)
        self.generated = 0

    def _call_api(self, prompt: str) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        body = {
            "model": TEACHER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 1024},
        }
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    OLLAMA_API_URL,
                    headers=headers,
                    json=body,
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except (requests.exceptions.RequestException, KeyError, IndexError):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        return None

    def generate_one(self, prompt: Optional[str] = None) -> Optional[str]:
        if prompt is None:
            prompt = self.rng.choice(PROBLEM_TEMPLATES)
        generated = self._call_api(prompt)
        if generated is not None:
            self.generated += 1
        return generated

    def generate_batch(
        self,
        n: int,
        custom_prompts: Optional[list[str]] = None,
    ) -> list[str]:
        samples = []

        for i in range(1, n + 1):
            if custom_prompts:
                prompt = custom_prompts[(i - 1) % len(custom_prompts)]
            else:
                prompt = self.rng.choice(PROBLEM_TEMPLATES)

            sample = self.generate_one(prompt)
            if sample is not None:
                samples.append(sample)

            if i % 10 == 0:
                print(f"Generated {i}/{n} samples")

        return samples

    def save_jsonl(self, samples: list[str], filename: Optional[str] = None) -> str:
        filename = filename or f"synthetic_{len(samples)}.jsonl"
        path = self.output_dir / filename

        with path.open("w", encoding="utf-8") as file:
            for sample in samples:
                record = {"content": sample, "source": "gpt-4o-mini"}
                file.write(json.dumps(record) + "\n")

        print(f"Saved {len(samples)} samples to {path}")
        return str(path)

    def generate_and_save(self, n: int, filename: Optional[str] = None) -> str:
        samples = self.generate_batch(n)
        return self.save_jsonl(samples, filename)

    def estimate_cost(self, n: int, avg_tokens_per_sample: int = 800) -> dict:
        return {
            "n_samples": n,
            "estimated_tokens": n * avg_tokens_per_sample,
            "cost_usd": 0.0,
            "note": "Free - running Qwen2.5-Coder-7B locally via Ollama",
        }


def check_synthetic(config: SamatNextConfig) -> None:
    gen = SyntheticDataGenerator(config)
    cost = gen.estimate_cost(500)
    print("SyntheticDataGenerator built")
    print(f"Teacher model: {TEACHER_MODEL}")
    print(f"Problem templates: {len(PROBLEM_TEMPLATES)}")
    print(f"Cost: {cost['note']}")
    print(f"Output dir: {gen.output_dir}")
    print("Synthetic OK")


if __name__ == "__main__":
    check_synthetic(DEFAULT_CONFIG)
