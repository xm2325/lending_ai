from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    sequence_length: int = 6
    sequence_dim: int = 5
    static_dim: int = 3
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 8
    top_fraction: float = 0.10
