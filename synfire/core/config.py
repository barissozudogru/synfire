"""Configuration dataclasses for all synfire components."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal


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
    # Mini-batch size for training. 0 means full-batch (all pairs at once).
    batch_size: int = 256
    # Early stopping: stop if loss improvement < min_delta for `patience` consecutive epochs.
    # Set patience=0 to disable early stopping.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1e-4
    # LR schedule: "none"/"constant" keeps lr fixed; "cosine" applies cosine decay;
    # "warmup_cosine" linearly warms up then applies cosine decay.
    lr_schedule: Literal["none", "constant", "cosine", "warmup_cosine"] = "none"
    # Fraction of total epochs used for linear warmup (only for "warmup_cosine").
    lr_warmup_fraction: float = 0.1
    # Gradient clipping: clip gradient norm to this value. 0.0 = disabled.
    grad_clip_norm: float = 0.0
    # Optimizer: "sgd" (default, backward compatible) or "adam".
    optimizer: Literal["sgd", "adam"] = "sgd"
    # Weight decay (L2 regularization) coefficient. 0.0 = disabled.
    weight_decay: float = 0.0
    # Layer normalization between linear transform and ReLU (Hinton's FF paper).
    layer_norm: bool = False
    # Negative sampling strategy: "random" (default), "hard", or "curriculum".
    negative_strategy: Literal["random", "hard", "curriculum"] = "random"

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
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be >= 0, got {self.early_stopping_patience}"
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta must be >= 0, got {self.early_stopping_min_delta}"
            )
        if self.lr_schedule not in ("none", "constant", "cosine", "warmup_cosine"):
            raise ValueError(
                f"lr_schedule must be 'none', 'constant', 'cosine', or 'warmup_cosine', "
                f"got {self.lr_schedule!r}"
            )
        if not (0.0 <= self.lr_warmup_fraction <= 1.0):
            raise ValueError(
                f"lr_warmup_fraction must be in [0, 1], got {self.lr_warmup_fraction}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(f"grad_clip_norm must be >= 0, got {self.grad_clip_norm}")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {self.optimizer!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.negative_strategy not in ("random", "hard", "curriculum"):
            raise ValueError(
                f"negative_strategy must be 'random', 'hard', or 'curriculum', "
                f"got {self.negative_strategy!r}"
            )


@dataclass(frozen=True)
class FFStackConfig:
    layer_dims: tuple[int, ...] = (64,)
    lr: float = 0.05
    threshold: float = 2.0
    epochs_per_layer: int = 30
    seed: int = 42
    # Mini-batch size forwarded to each FF layer (0 = full-batch).
    batch_size: int = 256
    # Early stopping patience forwarded to each FF layer (0 = disabled).
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1e-4
    # LR schedule applied to each layer.
    lr_schedule: Literal["none", "constant", "cosine", "warmup_cosine"] = "none"
    # Fraction of epochs for linear warmup (only for "warmup_cosine").
    lr_warmup_fraction: float = 0.1
    # Gradient clipping norm (0.0 = disabled).
    grad_clip_norm: float = 0.0
    # Optimizer forwarded to each FF layer. "sgd" or "adam".
    optimizer: Literal["sgd", "adam"] = "sgd"
    # Weight decay forwarded to each FF layer.
    weight_decay: float = 0.0
    # Layer normalization forwarded to each FF layer.
    layer_norm: bool = False
    # Negative sampling strategy forwarded to each FF layer.
    negative_strategy: Literal["random", "hard", "curriculum"] = "random"

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
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be >= 0, got {self.early_stopping_patience}"
            )
        if self.lr_schedule not in ("none", "constant", "cosine", "warmup_cosine"):
            raise ValueError(
                f"lr_schedule must be 'none', 'constant', 'cosine', or 'warmup_cosine', "
                f"got {self.lr_schedule!r}"
            )
        if not (0.0 <= self.lr_warmup_fraction <= 1.0):
            raise ValueError(
                f"lr_warmup_fraction must be in [0, 1], got {self.lr_warmup_fraction}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(f"grad_clip_norm must be >= 0, got {self.grad_clip_norm}")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {self.optimizer!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.negative_strategy not in ("random", "hard", "curriculum"):
            raise ValueError(
                f"negative_strategy must be 'random', 'hard', or 'curriculum', "
                f"got {self.negative_strategy!r}"
            )


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
    # When True, goodness is aggregated across ALL stack layers (weighted mean
    # with later layers having higher weight), not just the final layer.
    # This typically improves detection quality when the stack has >= 2 layers.
    ensemble_goodness: bool = False

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
    # When True, after FF training the goodness threshold is recalibrated to the
    # mean goodness of positive training samples. This makes the threshold
    # data-driven rather than relying on the initial config value.
    adaptive_threshold: bool = False

    def replace(self, **kwargs: Any) -> SynfireConfig:
        """Return a new SynfireConfig with the specified fields replaced.

        Supports nested paths using double-underscore notation. For example::

            config.replace(ff_stack__lr=0.01, hebbian__n_prototypes=8)

        Top-level fields can also be replaced directly::

            config.replace(adaptive_threshold=False)

        Args:
            **kwargs: Field replacements. Use ``parent__child=value`` to update
                a nested config object.

        Returns:
            New ``SynfireConfig`` with the requested changes applied.

        Raises:
            ValueError: If an unrecognized field path is provided.
        """
        # Gather direct (top-level) replacements and nested ones separately.
        direct: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}

        top_field_names = {f.name for f in dataclasses.fields(self)}

        for key, value in kwargs.items():
            if "__" in key:
                parent, child = key.split("__", 1)
                if parent not in top_field_names:
                    raise ValueError(
                        f"Unknown top-level field {parent!r}. "
                        f"Valid fields: {sorted(top_field_names)}"
                    )
                nested.setdefault(parent, {})[child] = value
            else:
                if key not in top_field_names:
                    raise ValueError(
                        f"Unknown field {key!r}. "
                        f"Valid fields: {sorted(top_field_names)}"
                    )
                direct[key] = value

        # Detect conflict: same parent used both as a direct replacement and as a nested path.
        conflict = set(direct) & set(nested)
        if conflict:
            raise ValueError(
                f"Field(s) {sorted(conflict)} appear in both direct and nested replacements. "
                "Use either 'parent=value' or 'parent__child=value', not both."
            )

        # Apply nested replacements by creating new sub-config instances.
        for parent_name, child_kwargs in nested.items():
            current_sub = getattr(self, parent_name)
            if not dataclasses.is_dataclass(current_sub):
                raise ValueError(
                    f"Cannot use nested path for non-dataclass field {parent_name!r}"
                )
            sub_field_names = {f.name for f in dataclasses.fields(current_sub)}
            for child_key in child_kwargs:
                if child_key not in sub_field_names:
                    raise ValueError(
                        f"Unknown field {parent_name}__{child_key!r}. "
                        f"Valid fields for {parent_name!r}: {sorted(sub_field_names)}"
                    )
            direct[parent_name] = dataclasses.replace(current_sub, **child_kwargs)

        return dataclasses.replace(self, **direct)
