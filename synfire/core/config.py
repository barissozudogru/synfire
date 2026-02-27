"""Configuration dataclasses for all synfire components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 25
    stride: int = 1

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")


@dataclass(frozen=True)
class NormConfig:
    method: Literal["zscore", "minmax"] = "zscore"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.method not in ("zscore", "minmax"):
            raise ValueError(f"method must be 'zscore' or 'minmax', got {self.method!r}")
        if self.eps < 0:
            raise ValueError(f"eps must be >= 0, got {self.eps}")


@dataclass(frozen=True)
class FFLayerConfig:
    input_dim: int = 50
    hidden_dim: int = 64
    lr: float = 0.05
    threshold: float = 2.0
    epochs: int = 30
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")


@dataclass(frozen=True)
class FFStackConfig:
    layer_dims: tuple[int, ...] = (64,)
    lr: float = 0.05
    threshold: float = 2.0
    epochs_per_layer: int = 30
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.layer_dims:
            raise ValueError("layer_dims must be non-empty")
        if any(d < 1 for d in self.layer_dims):
            raise ValueError(f"all layer_dims must be >= 1, got {self.layer_dims}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.epochs_per_layer < 1:
            raise ValueError(f"epochs_per_layer must be >= 1, got {self.epochs_per_layer}")


@dataclass(frozen=True)
class HebbianConfig:
    n_prototypes: int = 16
    lr: float = 0.01
    inhibition_strength: float = 0.1
    epochs: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_prototypes < 1:
            raise ValueError(f"n_prototypes must be >= 1, got {self.n_prototypes}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.inhibition_strength < 0:
            raise ValueError(
                f"inhibition_strength must be >= 0, got {self.inhibition_strength}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")


@dataclass(frozen=True)
class AnomalyConfig:
    weight_goodness: float = 0.2
    weight_distance: float = 0.6
    weight_transition: float = 0.2
    use_goodness: bool = True
    use_distance: bool = True
    use_transition: bool = True

    def __post_init__(self) -> None:
        if self.weight_goodness < 0:
            raise ValueError(f"weight_goodness must be >= 0, got {self.weight_goodness}")
        if self.weight_distance < 0:
            raise ValueError(f"weight_distance must be >= 0, got {self.weight_distance}")
        if self.weight_transition < 0:
            raise ValueError(f"weight_transition must be >= 0, got {self.weight_transition}")


@dataclass(frozen=True)
class SynfireConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    ff_stack: FFStackConfig = field(default_factory=FFStackConfig)
    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
