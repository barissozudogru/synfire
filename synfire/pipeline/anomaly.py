"""Anomaly scoring combining goodness, prototype distance, and transition surprise."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import AnomalyConfig
from synfire.layers.ff_layer import goodness
from synfire.layers.ff_stack import FFStackState, forward_stack
from synfire.layers.hebbian import HebbianState, assign, distances_to_prototypes

logger = logging.getLogger(__name__)


@dataclass
class DecomposedAnomalyScore:
    """Per-component anomaly scores with combined total.

    Attributes:
        goodness_deficit: Normalized goodness deficit component (batch,).
            None when use_goodness=False.
        prototype_distance: Normalized prototype distance component (batch,).
            None when use_distance=False.
        transition_surprise: Normalized transition surprise component (batch,).
            None when use_transition=False.
        combined: Weighted combined anomaly score (batch,). Same as anomaly_scores().
    """
    goodness_deficit: NDArray | None
    prototype_distance: NDArray | None
    transition_surprise: NDArray | None
    combined: NDArray


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


def _ensemble_goodness_deficit(
    activations: list[NDArray], threshold: float
) -> NDArray:
    """Compute goodness deficit aggregated across all stack layers.

    Later layers receive higher weight via a linearly increasing scheme,
    reflecting that deeper representations are more semantically meaningful.
    The deficit (threshold - goodness) is averaged across layers; higher values
    indicate inputs that fail to achieve the expected goodness level.

    Args:
        activations: List of per-layer activations, shape [(batch, d_i), ...].
        threshold: Target goodness threshold.

    Returns:
        Aggregated deficit of shape (batch,).
    """
    n_layers = len(activations)
    # Weights: 1, 2, ..., n_layers (later layers weighted more heavily)
    weights = np.arange(1, n_layers + 1, dtype=np.float64)
    weights = weights / weights.sum()

    deficit = np.zeros(activations[0].shape[0])
    for w, acts in zip(weights, activations):
        deficit += w * (threshold - goodness(acts))
    return deficit


def _compute_components(
    stack: FFStackState,
    hebbian: HebbianState,
    x: NDArray,
    config: AnomalyConfig,
    threshold: float,
    trans_prob: NDArray | None = None,
    activations: list[NDArray] | None = None,
) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
    """Compute the three raw scoring components (before normalization)."""
    if activations is None:
        activations = forward_stack(stack, x)
    representations = activations[-1]

    goodness_deficit: NDArray | None = None
    if config.use_goodness:
        if config.ensemble_goodness and len(activations) > 1:
            goodness_deficit = _ensemble_goodness_deficit(activations, threshold)
        else:
            goodness_deficit = threshold - goodness(representations)

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
    # Build training transition matrix (single forward pass, reused below)
    train_activations = forward_stack(stack, x_train)
    representations = train_activations[-1]
    labels = assign(hebbian, representations)
    train_trans_prob = _build_transition_matrix(labels, hebbian.config.n_prototypes)

    g_def, dist, surprise = _compute_components(
        stack, hebbian, x_train, config, threshold,
        trans_prob=train_trans_prob, activations=train_activations,
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


def anomaly_scores_decomposed(
    stack: FFStackState,
    hebbian: HebbianState,
    x: NDArray,
    config: AnomalyConfig | None = None,
    threshold: float = 2.0,
    scaler: AnomalyScaler | None = None,
) -> DecomposedAnomalyScore:
    """Compute decomposed anomaly scores returning each component separately.

    Returns individual normalized components (goodness_deficit, prototype_distance,
    transition_surprise) alongside the weighted combined score. Components not
    enabled in config are returned as None.

    Args:
        stack: Trained FF stack.
        hebbian: Trained Hebbian state.
        x: Input pairs of shape (batch, input_dim).
        config: Anomaly scoring weights.
        threshold: FF goodness threshold.
        scaler: Pre-computed normalization stats from training data.

    Returns:
        DecomposedAnomalyScore with per-component arrays and combined score.
    """
    if config is None:
        config = AnomalyConfig()

    trans_prob = scaler.trans_prob if scaler is not None else None
    g_def, dist, surprise = _compute_components(
        stack, hebbian, x, config, threshold, trans_prob=trans_prob,
    )

    combined = np.zeros(len(x))
    norm_g: NDArray | None = None
    norm_d: NDArray | None = None
    norm_s: NDArray | None = None

    if config.use_goodness and g_def is not None:
        if scaler is not None:
            norm_g = _normalize_fixed(g_def, scaler.goodness_min, scaler.goodness_range)
        else:
            norm_g = _normalize_batch(g_def)
        combined += config.weight_goodness * norm_g

    if config.use_distance and dist is not None:
        if scaler is not None:
            norm_d = _normalize_fixed(dist, scaler.distance_min, scaler.distance_range)
        else:
            norm_d = _normalize_batch(dist)
        combined += config.weight_distance * norm_d

    if config.use_transition and surprise is not None:
        if scaler is not None:
            norm_s = _normalize_fixed(surprise, scaler.surprise_min, scaler.surprise_range)
        else:
            norm_s = _normalize_batch(surprise)
        combined += config.weight_transition * norm_s

    return DecomposedAnomalyScore(
        goodness_deficit=norm_g,
        prototype_distance=norm_d,
        transition_surprise=norm_s,
        combined=combined,
    )
