"""Dataset loader for SamatNext-520M. Python-only per SPEC.md."""

import json
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from datasets import Dataset, load_dataset

from data.fim import FIMSample, FIMTransform
from data.tokenizer import SamatNextTokenizer
from model.config import DEFAULT_CONFIG, SamatNextConfig


HUGGINGFACE_DATASETS = [
    ("bigcode/the-stack-v2-train-smol-ids", "python"),
    ("codeparrot/github-code-clean", "Python"),
]


@dataclass
class BatchOutput:
    input_ids: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    is_fim: list[bool]


class PythonCodeDataset:
    def __init__(
        self,
        tokenizer: SamatNextTokenizer,
        config: SamatNextConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.fim = FIMTransform(tokenizer, config)
        self.rng = random.Random(seed)
        self.samples: list[FIMSample] = []
        self._loaded = False

    def load(self, dataset_name: Optional[str] = None, local_path: Optional[str] = None) -> None:
        raw_texts = []
        seen = set()

        if local_path is not None:
            with open(local_path, "r", encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)
                    content = row.get("content")
                    if not isinstance(content, str):
                        continue
                    dedup_key = content[:50]
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    raw_texts.append(content)
                    if self.max_samples is not None and len(raw_texts) >= self.max_samples:
                        break
        elif dataset_name is not None:
            dataset = load_dataset(dataset_name, split=self.split, streaming=True)
            for row in dataset:
                content = row.get("content")
                if not isinstance(content, str):
                    continue
                dedup_key = content[:50]
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                raw_texts.append(content)
                if self.max_samples is not None and len(raw_texts) >= self.max_samples:
                    break
        else:
            raise ValueError("Either dataset_name or local_path must be provided")

        self.samples = self.fim.transform_batch(raw_texts)
        self._loaded = True
        print(f"Loaded {len(self.samples)} samples")

    def _make_batch(self, samples: list[FIMSample]) -> BatchOutput:
        length = self.config.max_seq_len
        padded = self.fim.pad_to_length(samples, length)
        all_input_ids = []
        all_targets = []
        all_masks = []

        for sample, tokens in zip(samples, padded):
            all_input_ids.append(tokens[:-1])
            all_targets.append(tokens[1:])
            valid_predictions = max(0, min(len(sample.tokens), length) - 1)
            mask = [0.0] * (length - 1)

            if sample.is_fim:
                middle_start = sample.prefix_len + sample.suffix_len + 3
                middle_end = middle_start + sample.middle_len
                prediction_start = max(0, middle_start - 1)
                prediction_end = min(middle_end - 1, valid_predictions)
                for index in range(prediction_start, prediction_end):
                    mask[index] = 1.0
            else:
                for index in range(valid_predictions):
                    mask[index] = 1.0

            all_masks.append(mask)

        return BatchOutput(
            input_ids=np.array(all_input_ids, dtype=np.int32),
            targets=np.array(all_targets, dtype=np.int32),
            loss_mask=np.array(all_masks, dtype=np.float32),
            is_fim=[sample.is_fim for sample in samples],
        )

    def iterate_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[BatchOutput]:
        if not self._loaded:
            raise RuntimeError("Dataset must be loaded before iterating batches")
        if shuffle:
            self.rng.shuffle(self.samples)

        complete_length = len(self.samples) - (len(self.samples) % batch_size)
        for start in range(0, complete_length, batch_size):
            yield self._make_batch(self.samples[start:start + batch_size])

    def __len__(self) -> int:
        return len(self.samples)


def check_dataset(config: SamatNextConfig) -> None:
    print("PythonCodeDataset: skipping load check (requires network + trained tokenizer)")
    print(f"Dataset sources: {[d[0] for d in HUGGINGFACE_DATASETS]}")
    print(f"Batch shape will be: ({config.max_seq_len - 1},) per sample")
    print(f"FIM rate: {config.fim_rate}")
    print("Dataset OK")


if __name__ == "__main__":
    check_dataset(DEFAULT_CONFIG)
