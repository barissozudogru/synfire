"""Stacked Forward-Forward layers with greedy layer-wise training."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import FFLayerConfig, FFStackConfig
from synfire.layers.ff_layer import FFLayerState, forward, init_layer, train_layer

logger = logging.getLogger(__name__)


@dataclass
class FFStackState:
    layers: list[FFLayerState]
    config: FFStackConfig


def init_stack(input_dim: int, config: FFStackConfig) -> FFStackState:
    """Initialize a stack of FF layers.

    Each layer's input_dim is the previous layer's hidden_dim.
    """
    layers = []
    current_dim = input_dim
    rng = np.random.default_rng(config.seed)

    for _i, hidden_dim in enumerate(config.layer_dims):
        layer_cfg = FFLayerConfig(
            input_dim=current_dim,
            hidden_dim=hidden_dim,
            lr=config.lr,
            threshold=config.threshold,
            epochs=config.epochs_per_layer,
            seed=int(rng.integers(0, 2**31)),
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            lr_schedule=config.lr_schedule,
            lr_warmup_fraction=config.lr_warmup_fraction,
            grad_clip_norm=config.grad_clip_norm,
            optimizer=config.optimizer,
            weight_decay=config.weight_decay,
            layer_norm=config.layer_norm,
            negative_strategy=config.negative_strategy,
        )
        layers.append(init_layer(layer_cfg))
        current_dim = hidden_dim

    return FFStackState(layers=layers, config=config)


def train_stack(
    state: FFStackState,
    x_pos: NDArray,
    x_neg: NDArray,
) -> tuple[FFStackState, list[list[float]]]:
    """Train all layers greedily: layer i trains on activations from trained layer i-1.

    Args:
        state: Stack state.
        x_pos: Positive input pairs, shape (batch, input_dim).
        x_neg: Negative input pairs, shape (batch, input_dim).

    Returns:
        (trained_stack_state, per_layer_loss_histories).
    """
    all_losses = []
    trained_layers = []
    current_pos = x_pos
    current_neg = x_neg

    for i, layer in enumerate(state.layers):
        logger.info(
            "Training stack layer %d/%d (input_dim=%d, hidden_dim=%d)",
            i + 1,
            len(state.layers),
            layer.config.input_dim,
            layer.config.hidden_dim,
        )
        trained_layer, losses = train_layer(layer, current_pos, current_neg)
        trained_layers.append(trained_layer)
        all_losses.append(losses)
        logger.info(
            "Layer %d/%d done: %d epochs, loss %.4f -> %.4f",
            i + 1,
            len(state.layers),
            len(losses),
            losses[0] if losses else float("nan"),
            losses[-1] if losses else float("nan"),
        )

        # Transform inputs through the trained layer for the next one
        current_pos = forward(trained_layer, current_pos)
        current_neg = forward(trained_layer, current_neg)

    return FFStackState(layers=trained_layers, config=state.config), all_losses


def forward_stack(state: FFStackState, x: NDArray) -> list[NDArray]:
    """Forward pass through all layers, returning activations at each level.

    Returns:
        List of activations, one per layer.
    """
    activations = []
    current = x
    for layer in state.layers:
        current = forward(layer, current)
        activations.append(current)
    return activations


def extract_representation(state: FFStackState, x: NDArray) -> NDArray:
    """Extract the final layer's activations as the representation.

    Args:
        state: Trained stack state.
        x: Input of shape (batch, input_dim).

    Returns:
        Representation of shape (batch, last_hidden_dim).
    """
    current = x
    for layer in state.layers:
        current = forward(layer, current)
    return current
