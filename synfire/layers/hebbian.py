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

    for i in range(1, k):
        # Compute distances to nearest existing prototype
        dists = np.min(
            np.sum((data[:, np.newaxis, :] - prototypes[np.newaxis, :i, :]) ** 2, axis=2),
            axis=1,
        )
        # Sample proportional to distance squared
        probs = dists / dists.sum()
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
    """
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

    # Lateral inhibition: for each prototype, push away from non-assigned inputs
    for j in range(n_proto):
        non_winners = winners != j
        if not np.any(non_winners):
            continue
        repel_inputs = x[non_winners]  # (M, D)
        repel_vec = new_prototypes[j] - repel_inputs  # (M, D)
        norms = np.linalg.norm(repel_vec, axis=1, keepdims=True) + 1e-12
        repel_unit = repel_vec / norms
        mean_repel = repel_unit.mean(axis=0)
        new_prototypes[j] += inhibition * lr * mean_repel

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
