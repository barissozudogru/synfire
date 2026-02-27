"""Single Forward-Forward layer with local gradient training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synfire.core.backend import backend
from synfire.core.config import FFLayerConfig


@dataclass
class FFLayerState:
    W: NDArray  # (hidden_dim, input_dim)
    b: NDArray  # (hidden_dim,)
    config: FFLayerConfig


def init_layer(config: FFLayerConfig) -> FFLayerState:
    """Initialize a single FF layer with Xavier initialization."""
    rng = np.random.default_rng(config.seed)
    scale = np.sqrt(2.0 / (config.input_dim + config.hidden_dim))
    W = rng.standard_normal((config.hidden_dim, config.input_dim)) * scale
    b = np.zeros(config.hidden_dim)
    return FFLayerState(W=W, b=b, config=config)


def forward(state: FFLayerState, x: NDArray) -> NDArray:
    """Forward pass: h = relu(x @ W^T + b).

    Args:
        state: Layer state with weights and bias.
        x: Input of shape (batch, input_dim).

    Returns:
        Activations of shape (batch, hidden_dim).
    """
    pre = backend.matmul(x, state.W.T) + state.b
    return backend.relu(pre)


def goodness(activations: NDArray) -> NDArray:
    """Compute goodness as mean squared activation per sample.

    Args:
        activations: Shape (batch, hidden_dim).

    Returns:
        Goodness scores of shape (batch,).
    """
    return backend.mean(activations**2, axis=1)


def compute_loss(
    g_pos: NDArray, g_neg: NDArray, threshold: float
) -> tuple[float, NDArray, NDArray]:
    """Compute FF loss: -log(sig(g_pos - theta)) - log(sig(theta - g_neg)).

    Returns:
        (scalar_loss, d_loss_d_g_pos, d_loss_d_g_neg) for gradient computation.
    """
    sig_pos = backend.sigmoid(g_pos - threshold)
    sig_neg = backend.sigmoid(threshold - g_neg)

    loss = -backend.mean(backend.log(sig_pos)) - backend.mean(backend.log(sig_neg))

    # Gradients: d_loss/d_g_pos = -(1 - sig_pos) / batch_size
    #            d_loss/d_g_neg = (1 - sig_neg) / batch_size
    batch_size = len(g_pos)
    d_g_pos = -(1.0 - sig_pos) / batch_size
    d_g_neg = (1.0 - sig_neg) / batch_size

    return float(loss), d_g_pos, d_g_neg


def _backward_goodness(activations: NDArray, d_goodness: NDArray) -> NDArray:
    """Gradient of goodness wrt activations.

    goodness = mean(h^2) -> d_goodness/d_h = 2*h / hidden_dim
    """
    hidden_dim = activations.shape[1]
    return (2.0 * activations / hidden_dim) * d_goodness[:, np.newaxis]


def _backward_relu(pre_activation: NDArray, d_h: NDArray) -> NDArray:
    """Gradient through ReLU."""
    return d_h * (pre_activation > 0).astype(d_h.dtype)


def train_step(
    state: FFLayerState,
    x_pos: NDArray,
    x_neg: NDArray,
) -> tuple[FFLayerState, float]:
    """Single training step: forward both, compute loss, update weights.

    Args:
        state: Current layer state.
        x_pos: Positive input pairs, shape (batch, input_dim).
        x_neg: Negative input pairs, shape (batch, input_dim).

    Returns:
        (updated_state, loss_value).
    """
    # Forward
    pre_pos = backend.matmul(x_pos, state.W.T) + state.b
    h_pos = backend.relu(pre_pos)
    g_pos = goodness(h_pos)

    pre_neg = backend.matmul(x_neg, state.W.T) + state.b
    h_neg = backend.relu(pre_neg)
    g_neg = goodness(h_neg)

    # Loss
    loss_val, d_g_pos, d_g_neg = compute_loss(g_pos, g_neg, state.config.threshold)

    # Backward through goodness -> relu -> linear
    d_h_pos = _backward_goodness(h_pos, d_g_pos)
    d_pre_pos = _backward_relu(pre_pos, d_h_pos)

    d_h_neg = _backward_goodness(h_neg, d_g_neg)
    d_pre_neg = _backward_relu(pre_neg, d_h_neg)

    # Gradients for W and b
    dW = backend.matmul(d_pre_pos.T, x_pos) + backend.matmul(d_pre_neg.T, x_neg)
    db = np.sum(d_pre_pos, axis=0) + np.sum(d_pre_neg, axis=0)

    # Update
    new_W = state.W - state.config.lr * dW
    new_b = state.b - state.config.lr * db

    return FFLayerState(W=new_W, b=new_b, config=state.config), loss_val


def train_layer(
    state: FFLayerState,
    x_pos: NDArray,
    x_neg: NDArray,
) -> tuple[FFLayerState, list[float]]:
    """Train a layer for the configured number of epochs.

    Returns:
        (trained_state, loss_history).
    """
    losses = []
    for _ in range(state.config.epochs):
        state, loss = train_step(state, x_pos, x_neg)
        losses.append(loss)
    return state, losses
