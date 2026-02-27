"""Cluster assignment from Hebbian competitive layer."""

from __future__ import annotations

from numpy.typing import NDArray

from synfire.layers.hebbian import HebbianState, assign, distances_to_prototypes


def cluster_assign(state: HebbianState, representations: NDArray) -> NDArray:
    """Assign cluster labels to representations.

    Args:
        state: Trained Hebbian state.
        representations: Shape (batch, repr_dim).

    Returns:
        Cluster indices of shape (batch,).
    """
    return assign(state, representations)


def cluster_distances(state: HebbianState, representations: NDArray) -> NDArray:
    """Compute distance from each representation to its nearest prototype.

    Returns:
        Distances of shape (batch,).
    """
    return distances_to_prototypes(state, representations)
