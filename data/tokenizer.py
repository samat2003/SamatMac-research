"""Tokenizer wrapper for SamatNext-520M. Python-only BPE vocabulary per SPEC.md."""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from model.config import DEFAULT_CONFIG, SamatNextConfig


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
FIM_PREFIX_TOKEN = "<fim_prefix>"
FIM_SUFFIX_TOKEN = "<fim_suffix>"
FIM_MIDDLE_TOKEN = "<fim_middle>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    FIM_PREFIX_TOKEN,
    FIM_SUFFIX_TOKEN,
    FIM_MIDDLE_TOKEN,
]


class SamatNextTokenizer:
    def __init__(self, config: SamatNextConfig, tokenizer_path: str = None):
        self.config = config
        self.vocab_size = config.vocab_size
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = self._build_tokenizer()

    def _build_tokenizer(self) -> Tokenizer:
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        return tokenizer

    def train(self, files: list[str], save_path: str) -> None:
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(save_path)
        print(f"Tokenizer trained and saved to {save_path}")

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        return ids[: self.config.max_seq_len]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def batch_encode(self, texts: list[str], max_len: int = None) -> list[list[int]]:
        length = max_len if max_len is not None else self.config.max_seq_len
        encoded = []
        for text in texts:
            ids = self.encode(text)
            ids = ids[:length]
            ids = ids + [self.pad_id] * (length - len(ids))
            encoded.append(ids)
        return encoded

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(EOS_TOKEN)

    @property
    def fim_prefix_id(self) -> int:
        return self.tokenizer.token_to_id(FIM_PREFIX_TOKEN)

    @property
    def fim_suffix_id(self) -> int:
        return self.tokenizer.token_to_id(FIM_SUFFIX_TOKEN)

    @property
    def fim_middle_id(self) -> int:
        return self.tokenizer.token_to_id(FIM_MIDDLE_TOKEN)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str, config: SamatNextConfig = DEFAULT_CONFIG):
        instance = cls(config, tokenizer_path=path)
        if instance.tokenizer.decoder is None:
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            instance.tokenizer.decoder = ByteLevelDecoder()
        return instance


def check_tokenizer(config: SamatNextConfig) -> None:
    tok = SamatNextTokenizer(config)
    print("SamatNextTokenizer built (untrained)")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print(f"Vocab size target: {config.vocab_size}")
    print("Tokenizer OK")


if __name__ == "__main__":
    check_tokenizer(DEFAULT_CONFIG)
