"""Configuration dataclasses for all synfire components."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 20
    stride: int = 1


@dataclass(frozen=True)
class NormConfig:
    method: str = "zscore"  # "zscore" or "minmax"
    eps: float = 1e-8


@dataclass(frozen=True)
class FFLayerConfig:
    input_dim: int = 40
    hidden_dim: int = 64
    lr: float = 0.01
    threshold: float = 2.0
    epochs: int = 100
    seed: int = 42


@dataclass(frozen=True)
class FFStackConfig:
    layer_dims: tuple[int, ...] = (64, 32)
    lr: float = 0.01
    threshold: float = 2.0
    epochs_per_layer: int = 100
    seed: int = 42


@dataclass(frozen=True)
class HebbianConfig:
    n_prototypes: int = 8
    lr: float = 0.01
    inhibition_strength: float = 0.1
    epochs: int = 50
    seed: int = 42


@dataclass(frozen=True)
class AnomalyConfig:
    weight_goodness: float = 0.5
    weight_distance: float = 0.3
    weight_transition: float = 0.2


@dataclass(frozen=True)
class SynfireConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    ff_stack: FFStackConfig = field(default_factory=FFStackConfig)
    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
