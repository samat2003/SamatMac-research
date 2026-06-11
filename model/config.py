"""SamatNext-520M model configuration. All values locked to SPEC.md."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class SamatNextConfig:
    model_name: str = "SamatNext-520M"
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ffn: int = 4096
    vocab_size: int = 32000
    max_seq_len: int = 4096
    dropout: float = 0.0
    dtype: str = "bfloat16"
    delta_chunk_size: int = 64
    delta_sparse_gate: bool = True
    memory_bus_layers: int = 4
    diff_attn_lambda_init: float = 0.8
    rope_base: float = 10000.0
    yarn_scale: float = 1.0
    yarn_alpha: float = 1.0
    n_latent_tokens: int = 4
    fim_rate: float = 0.5
    mtp_heads: int = 6
    mtp_confidence_gate: bool = True
    mtp_calibration_weight: float = 0.1
    tie_embeddings: bool = False
    warmup_steps: int = 500
    layer_pattern: str = "gated_attention,diff"

    def __post_init__(self) -> None:
        if self.n_layers % 2 != 0:
            raise ValueError("n_layers must be even")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    def get_layer_type(self, layer_idx: int) -> str:
        return "gated_attention" if layer_idx % 2 == 0 else "diff"

    @property
    def num_gated_attention_layers(self) -> int:
        return self.n_layers // 2

    @property
    def num_diff_layers(self) -> int:
        return self.n_layers // 2

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv_head(self) -> int:
        return self.d_model // self.n_kv_heads

    @property
    def memory_bus_start_layer(self) -> int:
        return self.n_layers - 2 * self.memory_bus_layers

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamatNextConfig":
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = SamatNextConfig()
