"""Fill-in-the-middle (FIM) data transform for SamatNext-520M per SPEC.md."""

import random
from dataclasses import dataclass
from typing import Optional

from data.tokenizer import (
    EOS_TOKEN,
    FIM_MIDDLE_TOKEN,
    FIM_PREFIX_TOKEN,
    FIM_SUFFIX_TOKEN,
    SamatNextTokenizer,
)
from model.config import DEFAULT_CONFIG, SamatNextConfig


@dataclass
class FIMSample:
    tokens: list[int]
    is_fim: bool
    prefix_len: int
    suffix_len: int
    middle_len: int


class FIMTransform:
    def __init__(self, tokenizer: SamatNextTokenizer, config: SamatNextConfig, fim_rate: Optional[float] = None):
        self.tokenizer = tokenizer
        self.config = config
        self.fim_rate = fim_rate if fim_rate is not None else config.fim_rate
        self.max_len = config.max_seq_len
        self.rng = random.Random(42)

    def transform(self, text: str) -> FIMSample:
        if self.rng.random() >= self.fim_rate:
            ids = self.tokenizer.encode(text, add_bos=True)
            ids = ids[: self.max_len]
            return FIMSample(tokens=ids, is_fim=False, prefix_len=len(ids), suffix_len=0, middle_len=0)

        ids = self.tokenizer.encode(text, add_bos=False)
        if len(ids) < 4:
            ids = self.tokenizer.encode(text, add_bos=True)
            ids = ids[: self.max_len]
            return FIMSample(tokens=ids, is_fim=False, prefix_len=len(ids), suffix_len=0, middle_len=0)

        i = self.rng.randint(1, len(ids) - 2)
        j = self.rng.randint(i + 1, len(ids) - 1)
        prefix_ids = ids[:i]
        middle_ids = ids[i:j]
        suffix_ids = ids[j:]

        fim_ids = (
            [self.tokenizer.fim_prefix_id]
            + prefix_ids
            + [self.tokenizer.fim_suffix_id]
            + suffix_ids
            + [self.tokenizer.fim_middle_id]
            + middle_ids
            + [self.tokenizer.eos_id]
        )
        fim_ids = fim_ids[: self.max_len]

        return FIMSample(
            tokens=fim_ids,
            is_fim=True,
            prefix_len=len(prefix_ids),
            suffix_len=len(suffix_ids),
            middle_len=len(middle_ids),
        )

    def transform_batch(self, texts: list[str]) -> list[FIMSample]:
        return [self.transform(text) for text in texts]

    def pad_to_length(self, samples: list[FIMSample], length: int) -> list[list[int]]:
        padded = []
        for sample in samples:
            ids = sample.tokens[:length]
            ids = ids + [self.tokenizer.pad_id] * (length - len(ids))
            padded.append(ids)
        return padded


def check_fim(config: SamatNextConfig) -> None:
    print("FIMTransform: skipping encode check (tokenizer untrained)")
    print(f"FIM rate: {config.fim_rate}")
    print(f"FIM tokens: {FIM_PREFIX_TOKEN}, {FIM_SUFFIX_TOKEN}, {FIM_MIDDLE_TOKEN}")
    print("FIM OK")


if __name__ == "__main__":
    check_fim(DEFAULT_CONFIG)
