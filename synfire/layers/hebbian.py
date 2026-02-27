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

    for i in range(len(x)):
        w = winners[i]
        diff = x[i] - new_prototypes[w]
        # Hebbian: pull winner toward input
        new_prototypes[w] += lr * diff

        # Lateral inhibition: push non-winners away
        for j in range(state.config.n_prototypes):
            if j != w:
                repel = new_prototypes[j] - x[i]
                new_prototypes[j] += inhibition * lr * repel / (np.linalg.norm(repel) + 1e-12)

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
