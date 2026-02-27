"""Anomaly scoring combining goodness, prototype distance, and transition surprise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import AnomalyConfig
from synfire.layers.ff_layer import goodness
from synfire.layers.ff_stack import FFStackState, forward_stack
from synfire.layers.hebbian import HebbianState, assign, distances_to_prototypes


@dataclass
class AnomalyScaler:
    """Fixed normalization stats computed from training data."""

    goodness_min: float
    goodness_range: float
    distance_min: float
    distance_range: float
    surprise_min: float
    surprise_range: float
    trans_prob: NDArray  # (n_prototypes, n_prototypes) transition probability matrix


def _build_transition_matrix(
    labels: NDArray, n_prototypes: int, eps: float = 1e-12
) -> NDArray:
    """Build normalized transition probability matrix from label sequence."""
    from_labels = labels[:-1]
    to_labels = labels[1:]

    trans = np.zeros((n_prototypes, n_prototypes))
    np.add.at(trans, (from_labels, to_labels), 1)

    row_sums = trans.sum(axis=1, keepdims=True)
    return trans / (row_sums + eps)


def _transition_surprise(
    labels: NDArray, n_prototypes: int, eps: float = 1e-12,
    trans_prob: NDArray | None = None,
) -> NDArray:
    """Compute transition surprise based on cluster transition probabilities.

    If trans_prob is provided, uses that fixed matrix. Otherwise builds one
    from the label sequence (legacy batch-dependent behavior).

    Returns:
        Surprise scores of shape (len(labels),). First element is 0.
    """
    if trans_prob is None:
        trans_prob = _build_transition_matrix(labels, n_prototypes, eps)

    from_labels = labels[:-1]
    to_labels = labels[1:]

    surprise = np.empty(len(labels))
    surprise[0] = 0.0
    surprise[1:] = -np.log(trans_prob[from_labels, to_labels] + eps)

    return surprise


def _normalize_fixed(arr: NDArray, arr_min: float, arr_range: float) -> NDArray:
    """Normalize using pre-computed min/range."""
    if arr_range < 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - arr_min) / arr_range, 0.0, 1.0)


def _normalize_batch(arr: NDArray) -> NDArray:
    """Fallback batch normalization (legacy behavior)."""
    rng = arr.max() - arr.min()
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.min()) / rng


def _compute_components(
    stack: FFStackState,
    hebbian: HebbianState,
    x: NDArray,
    config: AnomalyConfig,
    threshold: float,
    trans_prob: NDArray | None = None,
) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
    """Compute the three raw scoring components (before normalization)."""
    activations = forward_stack(stack, x)
    representations = activations[-1]

    goodness_deficit = threshold - goodness(representations) if config.use_goodness else None
    dist = distances_to_prototypes(hebbian, representations) if config.use_distance else None

    surprise = None
    if config.use_transition:
        labels = assign(hebbian, representations)
        surprise = _transition_surprise(
            labels, hebbian.config.n_prototypes, trans_prob=trans_prob,
        )

    return goodness_deficit, dist, surprise


def fit_anomaly_scaler(
    stack: FFStackState,
    hebbian: HebbianState,
    x_train: NDArray,
    config: AnomalyConfig,
    threshold: float,
) -> AnomalyScaler:
    """Compute normalization statistics from training data.

    Args:
        stack: Trained FF stack.
        hebbian: Trained Hebbian state.
        x_train: Training pairs of shape (batch, input_dim).
        config: Anomaly scoring config.
        threshold: FF goodness threshold.

    Returns:
        AnomalyScaler with fixed normalization stats and transition matrix.
    """
    # Build training transition matrix
    activations = forward_stack(stack, x_train)
    representations = activations[-1]
    labels = assign(hebbian, representations)
    train_trans_prob = _build_transition_matrix(labels, hebbian.config.n_prototypes)

    g_def, dist, surprise = _compute_components(
        stack, hebbian, x_train, config, threshold, trans_prob=train_trans_prob,
    )

    def _stats(arr: NDArray | None) -> tuple[float, float]:
        if arr is None:
            return 0.0, 0.0
        return float(arr.min()), float(arr.max() - arr.min())

    g_min, g_range = _stats(g_def)
    d_min, d_range = _stats(dist)
    s_min, s_range = _stats(surprise)

    return AnomalyScaler(
        goodness_min=g_min,
        goodness_range=g_range,
        distance_min=d_min,
        distance_range=d_range,
        surprise_min=s_min,
        surprise_range=s_range,
        trans_prob=train_trans_prob,
    )


def anomaly_scores(
    stack: FFStackState,
    hebbian: HebbianState,
    x: NDArray,
    config: AnomalyConfig | None = None,
    threshold: float = 2.0,
    scaler: AnomalyScaler | None = None,
) -> NDArray:
    """Compute combined anomaly scores.

    score = w1 * norm(threshold - goodness) + w2 * norm(distance) + w3 * norm(surprise)

    When a scaler is provided, normalization uses fixed training statistics
    instead of the current batch's min/max, ensuring deterministic scoring.

    Args:
        stack: Trained FF stack.
        hebbian: Trained Hebbian state.
        x: Input pairs of shape (batch, input_dim).
        config: Anomaly scoring weights.
        threshold: FF goodness threshold.
        scaler: Pre-computed normalization stats from training data.

    Returns:
        Anomaly scores of shape (batch,). Higher = more anomalous.
    """
    if config is None:
        config = AnomalyConfig()

    trans_prob = scaler.trans_prob if scaler is not None else None
    g_def, dist, surprise = _compute_components(
        stack, hebbian, x, config, threshold, trans_prob=trans_prob,
    )

    score = np.zeros(len(x))

    if config.use_goodness and g_def is not None:
        if scaler is not None:
            score += config.weight_goodness * _normalize_fixed(
                g_def, scaler.goodness_min, scaler.goodness_range
            )
        else:
            score += config.weight_goodness * _normalize_batch(g_def)

    if config.use_distance and dist is not None:
        if scaler is not None:
            score += config.weight_distance * _normalize_fixed(
                dist, scaler.distance_min, scaler.distance_range
            )
        else:
            score += config.weight_distance * _normalize_batch(dist)

    if config.use_transition and surprise is not None:
        if scaler is not None:
            score += config.weight_transition * _normalize_fixed(
                surprise, scaler.surprise_min, scaler.surprise_range
            )
        else:
            score += config.weight_transition * _normalize_batch(surprise)

    return score
