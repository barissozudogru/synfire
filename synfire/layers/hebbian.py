"""Hebbian competitive learning layer with WTA and lateral inhibition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import HebbianConfig


@dataclass
class HebbianState:
    prototypes: NDArray  # (n_prototypes, input_dim)
    config: HebbianConfig


def _kmeans_plus_plus_init(
    data: NDArray, k: int, rng: np.random.Generator
) -> NDArray:
    """Initialize prototypes using k-means++ strategy."""
    n, d = data.shape
    prototypes = np.empty((k, d))

    # First prototype: random data point
    idx = rng.integers(0, n)
    prototypes[0] = data[idx]

    min_dists = np.full(n, np.inf)
    for i in range(1, k):
        new_dists = np.sum((data - prototypes[i - 1]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        total = min_dists.sum()
        probs = np.ones(n) / n if total < 1e-12 else min_dists / total
        idx = rng.choice(n, p=probs)
        prototypes[i] = data[idx]

    return prototypes


def init_hebbian(data: NDArray, config: HebbianConfig) -> HebbianState:
    """Initialize Hebbian layer with k-means++ prototype initialization.

    Args:
        data: Training data of shape (N, D) for initialization.
        config: Hebbian config.

    Returns:
        Initialized HebbianState.

    Raises:
        ValueError: If n_prototypes exceeds the number of data points.
    """
    if config.n_prototypes > len(data):
        raise ValueError(
            f"n_prototypes ({config.n_prototypes}) must not exceed the number of "
            f"data points ({len(data)}). Reduce n_prototypes or provide more training data."
        )
    rng = np.random.default_rng(config.seed)
    prototypes = _kmeans_plus_plus_init(data, config.n_prototypes, rng)
    return HebbianState(prototypes=prototypes, config=config)


def assign(state: HebbianState, x: NDArray) -> NDArray:
    """Winner-Take-All assignment: each input goes to nearest prototype.

    Args:
        state: Hebbian state.
        x: Input of shape (batch, D).

    Returns:
        Cluster indices of shape (batch,).
    """
    # (batch, n_prototypes)
    dists = np.sum(
        (x[:, np.newaxis, :] - state.prototypes[np.newaxis, :, :]) ** 2, axis=2
    )
    return np.argmin(dists, axis=1)


def distances_to_prototypes(state: HebbianState, x: NDArray) -> NDArray:
    """Compute distance from each input to its nearest prototype.

    Returns:
        Distances of shape (batch,).
    """
    dists = np.sum(
        (x[:, np.newaxis, :] - state.prototypes[np.newaxis, :, :]) ** 2, axis=2
    )
    return np.min(dists, axis=1)


def update_step(state: HebbianState, x: NDArray) -> HebbianState:
    """Single Hebbian update: move winner toward input, push losers away.

    Args:
        state: Current state.
        x: Single batch of inputs, shape (batch, D).

    Returns:
        Updated state.
    """
    winners = assign(state, x)
    new_prototypes = state.prototypes.copy()
    lr = state.config.lr
    inhibition = state.config.inhibition_strength
    n_proto = state.config.n_prototypes
    batch_size = len(x)

    # Winner attraction: compute per-prototype mean displacement
    one_hot = np.zeros((batch_size, n_proto))
    one_hot[np.arange(batch_size), winners] = 1.0
    counts = one_hot.sum(axis=0)  # (n_proto,)
    # Sum of (x_i - prototype_j) for samples assigned to j
    # x weighted by assignment: (n_proto, D)
    weighted_sum = one_hot.T @ x  # (n_proto, D)
    active = counts > 0
    mean_input = np.zeros_like(new_prototypes)
    mean_input[active] = weighted_sum[active] / counts[active, np.newaxis]
    displacement = mean_input - new_prototypes
    new_prototypes[active] += lr * displacement[active]

    # Lateral inhibition (vectorized): push each prototype away from non-assigned inputs.
    # assignment_mask[j, i] = True if sample i was NOT won by prototype j.
    # Shape: (n_proto, batch_size)
    assignment_mask = (np.arange(n_proto)[:, np.newaxis] != winners[np.newaxis, :])

    # Pairwise difference: proto_j - x_i for all (j, i).
    # new_prototypes: (n_proto, D) -> (n_proto, 1, D)
    # x:              (batch, D)  -> (1, batch, D)
    # repel_vecs:     (n_proto, batch, D)
    repel_vecs = new_prototypes[:, np.newaxis, :] - x[np.newaxis, :, :]

    # Normalize to unit vectors
    norms = np.linalg.norm(repel_vecs, axis=2, keepdims=True) + 1e-12
    repel_units = repel_vecs / norms  # (n_proto, batch, D)

    # Zero out winner contributions using the assignment mask
    # mask: (n_proto, batch, 1) broadcast over D
    mask = assignment_mask[:, :, np.newaxis].astype(repel_units.dtype)
    masked_repel = repel_units * mask  # (n_proto, batch, D)

    # Denominator: number of non-winner samples per prototype (avoid div-by-zero)
    n_non_winners = assignment_mask.sum(axis=1, keepdims=True)[:, :, np.newaxis]  # (n_proto,1,1)
    n_non_winners = np.maximum(n_non_winners, 1)

    # Mean repulsion vector per prototype (n_proto, D)
    mean_repel = masked_repel.sum(axis=1) / n_non_winners.squeeze(axis=(1, 2))[:, np.newaxis]

    new_prototypes += inhibition * lr * mean_repel

    return HebbianState(prototypes=new_prototypes, config=state.config)


def train_hebbian(
    state: HebbianState, data: NDArray, batch_size: int = 64
) -> HebbianState:
    """Train Hebbian layer for configured epochs.

    Args:
        state: Initial state.
        data: Training data of shape (N, D).
        batch_size: Mini-batch size for updates.

    Returns:
        Trained state.
    """
    rng = np.random.default_rng(state.config.seed)
    n = len(data)

    for _ in range(state.config.epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = data[indices[start : start + batch_size]]
            state = update_step(state, batch)

    return state
