"""Anomaly scoring combining goodness, prototype distance, and transition surprise."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import AnomalyConfig
from synfire.layers.ff_layer import goodness
from synfire.layers.ff_stack import FFStackState, forward_stack
from synfire.layers.hebbian import HebbianState, assign, distances_to_prototypes


def _goodness_scores(stack: FFStackState, x: NDArray) -> NDArray:
    """Compute goodness from the last FF layer's activations."""
    activations = forward_stack(stack, x)
    return goodness(activations[-1])


def _transition_surprise(
    labels: NDArray, n_prototypes: int, eps: float = 1e-12
) -> NDArray:
    """Compute transition surprise based on cluster transition probabilities.

    Estimates P(label[t+1] | label[t]) from the sequence, then returns
    -log(P) as surprise for each transition.

    Returns:
        Surprise scores of shape (len(labels),). First element is 0.
    """
    # Build transition count matrix
    trans = np.zeros((n_prototypes, n_prototypes))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1

    # Normalize to probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = trans / (row_sums + eps)

    # Compute surprise for each transition
    surprise = np.zeros(len(labels))
    for i in range(1, len(labels)):
        p = trans_prob[labels[i - 1], labels[i]]
        surprise[i] = -np.log(p + eps)

    return surprise


def anomaly_scores(
    stack: FFStackState,
    hebbian: HebbianState,
    x: NDArray,
    config: AnomalyConfig | None = None,
    threshold: float = 2.0,
) -> NDArray:
    """Compute combined anomaly scores.

    score = w1 * (threshold - goodness) + w2 * distance_to_prototype + w3 * transition_surprise

    Args:
        stack: Trained FF stack.
        hebbian: Trained Hebbian state.
        x: Input pairs of shape (batch, input_dim).
        config: Anomaly scoring weights.
        threshold: FF goodness threshold.

    Returns:
        Anomaly scores of shape (batch,). Higher = more anomalous.
    """
    if config is None:
        config = AnomalyConfig()

    # Component 1: Goodness deficit
    g = _goodness_scores(stack, x)
    goodness_deficit = threshold - g

    # Component 2: Distance to nearest prototype
    activations = forward_stack(stack, x)
    representations = activations[-1]
    dist = distances_to_prototypes(hebbian, representations)

    # Component 3: Transition surprise
    labels = assign(hebbian, representations)
    surprise = _transition_surprise(labels, hebbian.config.n_prototypes)

    # Normalize each component to [0, 1] range for balanced combination
    def _normalize(arr: NDArray) -> NDArray:
        rng = arr.max() - arr.min()
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr.min()) / rng

    score = (
        config.weight_goodness * _normalize(goodness_deficit)
        + config.weight_distance * _normalize(dist)
        + config.weight_transition * _normalize(surprise)
    )

    return score
