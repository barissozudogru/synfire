"""Analysis utilities: per-layer decomposition and prototype utilization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synfire.layers.ff_layer import goodness
from synfire.layers.ff_stack import FFStackState, forward_stack
from synfire.layers.hebbian import HebbianState, assign


def layer_anomaly_decomposition(
    stack: FFStackState,
    x: NDArray,
    threshold: float | None = None,
) -> NDArray:
    """Compute per-layer goodness deficit for each input sample.

    The deficit is ``threshold - goodness(activations)``. A positive value
    means the layer's activations failed to reach the threshold (anomalous).

    Args:
        stack: Trained FF stack.
        x: Input array of shape (batch, input_dim).
        threshold: Goodness threshold. Uses the first layer's config threshold
            when None.

    Returns:
        Array of shape (batch, n_layers) where entry [i, j] is the goodness
        deficit for sample i at layer j.
    """
    if threshold is None:
        threshold = stack.layers[0].config.threshold

    activations = forward_stack(stack, x)
    deficits = np.stack(
        [threshold - goodness(acts) for acts in activations], axis=1
    )
    return deficits


@dataclass
class LayerStats:
    """Per-layer activation statistics.

    Attributes:
        layer_index: Zero-based layer index.
        mean_activation: Mean activation value across all units and samples.
        std_activation: Standard deviation of activations.
        sparsity: Fraction of zero (dead) activations (ReLU produces zeros
            for negative pre-activations).
        mean_goodness: Mean goodness score across samples.
        std_goodness: Standard deviation of goodness scores.
    """

    layer_index: int
    mean_activation: float
    std_activation: float
    sparsity: float
    mean_goodness: float
    std_goodness: float


def activation_statistics(
    stack: FFStackState,
    x: NDArray,
) -> list[LayerStats]:
    """Compute per-layer activation statistics for a batch of inputs.

    Args:
        stack: Trained FF stack.
        x: Input array of shape (batch, input_dim).

    Returns:
        List of ``LayerStats``, one per layer.
    """
    activations = forward_stack(stack, x)
    stats = []
    for i, acts in enumerate(activations):
        g = goodness(acts)
        stats.append(
            LayerStats(
                layer_index=i,
                mean_activation=float(acts.mean()),
                std_activation=float(acts.std()),
                sparsity=float((acts == 0.0).mean()),
                mean_goodness=float(g.mean()),
                std_goodness=float(g.std()),
            )
        )
    return stats


def prototype_utilization(
    hebbian: HebbianState,
    representations: NDArray,
) -> NDArray:
    """Count how many representations are assigned to each prototype.

    Args:
        hebbian: Trained Hebbian state.
        representations: Array of shape (batch, repr_dim).

    Returns:
        Integer array of shape (n_prototypes,) with assignment counts.
    """
    labels = assign(hebbian, representations)
    counts = np.bincount(labels, minlength=hebbian.config.n_prototypes)
    return counts
