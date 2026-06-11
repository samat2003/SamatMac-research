"""Configurations for the 120M validation experiment."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Experiment120MConfig:
    """Shared base configuration for 120M models."""
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ffn: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 512
    dropout: float = 0.0
    dtype: str = "bfloat16"
    rope_base: float = 10000.0
    yarn_scale: float = 1.0
    yarn_alpha: float = 1.0
    fim_rate: float = 0.5
    tie_embeddings: bool = True
    warmup_steps: int = 500

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv_head(self) -> int:
        return self.d_model // self.n_kv_heads

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment120MConfig":
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Baseline120MConfig(Experiment120MConfig):
    """Configuration specific to the standard Transformer baseline."""
    model_name: str = "Baseline-120M"
    d_ffn: int = 3310  # Wider FFN to match SamatNext's parameter count


@dataclass
class SamatNext120MConfig(Experiment120MConfig):
    """Configuration specific to the scaled SamatNext architecture."""
    model_name: str = "SamatNext-120M"
    d_ffn: int = 2700  # Wider FFN to compensate for fewer layers
    delta_chunk_size: int = 64
    delta_sparse_gate: bool = True
    memory_bus_layers: int = 2
    diff_attn_lambda_init: float = 0.8
    n_latent_tokens: int = 4
    mtp_heads: int = 2
    mtp_confidence_gate: bool = True
    mtp_calibration_weight: float = 0.1
    layer_pattern: str = "gated_attention,diff"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_layers % 2 != 0:
            raise ValueError("SamatNext requires even n_layers")

    def get_layer_type(self, layer_idx: int) -> str:
        return "gated_attention" if layer_idx % 2 == 0 else "diff"

    @property
    def num_gated_attention_layers(self) -> int:
        return self.n_layers // 2

    @property
    def num_diff_layers(self) -> int:
        return self.n_layers // 2

    @property
    def memory_bus_start_layer(self) -> int:
        return self.n_layers - 2 * self.memory_bus_layers
