"""Representation extraction from FF stack."""

from __future__ import annotations

from numpy.typing import NDArray

from synfire.layers.ff_stack import FFStackState, extract_representation, forward_stack


def get_representation(state: FFStackState, x: NDArray) -> NDArray:
    """Extract final-layer representation from trained FF stack.

    Args:
        state: Trained stack state.
        x: Input pairs of shape (batch, input_dim).

    Returns:
        Representation of shape (batch, last_hidden_dim).
    """
    return extract_representation(state, x)


def get_all_layer_activations(state: FFStackState, x: NDArray) -> list[NDArray]:
    """Get activations at every layer for analysis.

    Returns:
        List of activations, one per layer.
    """
    return forward_stack(state, x)
