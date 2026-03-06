"""Single Forward-Forward layer with local gradient training."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from synfire.core.backend import backend
from synfire.core.config import FFLayerConfig

if TYPE_CHECKING:
    from synfire.callbacks import TrainingCallback

logger = logging.getLogger(__name__)


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
    lr: float | None = None,
) -> tuple[FFLayerState, float]:
    """Single training step: forward both, compute loss, update weights.

    Args:
        state: Current layer state.
        x_pos: Positive input pairs, shape (batch, input_dim).
        x_neg: Negative input pairs, shape (batch, input_dim).
        lr: Learning rate override. Uses state.config.lr when None.

    Returns:
        (updated_state, loss_value).
    """
    effective_lr = lr if lr is not None else state.config.lr

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

    # Gradient clipping by global norm
    clip_norm = state.config.grad_clip_norm
    if clip_norm > 0.0:
        grad_norm = math.sqrt(float(np.sum(dW ** 2)) + float(np.sum(db ** 2)))
        if grad_norm > clip_norm:
            scale = clip_norm / (grad_norm + 1e-12)
            dW = dW * scale
            db = db * scale

    # Update
    new_W = state.W - effective_lr * dW
    new_b = state.b - effective_lr * db

    if not np.all(np.isfinite(new_W)) or not np.all(np.isfinite(new_b)):
        raise RuntimeError(
            "Non-finite values detected in layer weights after update "
            "(NaN or Inf). Try reducing the learning rate or clipping gradients."
        )

    return FFLayerState(W=new_W, b=new_b, config=state.config), loss_val


def _scheduled_lr(
    base_lr: float,
    epoch: int,
    total_epochs: int,
    schedule: str,
    warmup_fraction: float = 0.1,
    lr_min: float = 0.0,
) -> float:
    """Compute the learning rate for a given epoch according to the schedule.

    Args:
        base_lr: Peak learning rate.
        epoch: Current epoch index (0-based).
        total_epochs: Total number of training epochs.
        schedule: One of "none", "constant", "cosine", "warmup_cosine".
        warmup_fraction: Fraction of total_epochs used for linear warmup
            (only relevant for "warmup_cosine").
        lr_min: Minimum learning rate for cosine schedules.

    Returns:
        Learning rate for this epoch.
    """
    if schedule in ("none", "constant") or total_epochs <= 1:
        return base_lr

    if schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * epoch / (total_epochs - 1)))
        return lr_min + (base_lr - lr_min) * decay

    if schedule == "warmup_cosine":
        warmup_epochs = 0 if warmup_fraction == 0.0 else max(1, int(total_epochs * warmup_fraction))
        if epoch < warmup_epochs:
            # Linear warmup from lr_min to base_lr
            return lr_min + (base_lr - lr_min) * (epoch + 1) / warmup_epochs
        # Cosine decay over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (base_lr - lr_min) * decay

    return base_lr


def train_layer(
    state: FFLayerState,
    x_pos: NDArray,
    x_neg: NDArray,
    callbacks: list[TrainingCallback] | None = None,
) -> tuple[FFLayerState, list[float]]:
    """Train a layer for the configured number of epochs with mini-batch SGD.

    Shuffles indices each epoch and processes data in mini-batches of size
    config.batch_size (0 = full batch). Applies the configured LR schedule
    and early stopping.

    Args:
        state: Initial layer state.
        x_pos: Positive input pairs, shape (batch, input_dim).
        x_neg: Negative input pairs, shape (batch, input_dim).
        callbacks: Optional list of :class:`~synfire.callbacks.TrainingCallback`
            objects that receive training events. Each callback's
            ``on_train_begin``, ``on_epoch_end``, and ``on_train_end`` methods
            are called at the appropriate points in the loop.

    Returns:
        (trained_state, loss_history). The history may be shorter than
        config.epochs when early stopping triggers.
    """
    config = state.config
    patience = config.early_stopping_patience
    min_delta = config.early_stopping_min_delta
    schedule = config.lr_schedule
    warmup_fraction = config.lr_warmup_fraction
    batch_size = config.batch_size
    n = len(x_pos)
    # batch_size=0 means full batch
    effective_batch = n if batch_size == 0 else min(batch_size, n)

    rng = np.random.default_rng(config.seed)

    cbs: list[TrainingCallback] = callbacks if callbacks is not None else []

    for cb in cbs:
        if hasattr(cb, 'on_train_begin'):
            cb.on_train_begin(total_epochs=config.epochs, n_samples=n)

    losses: list[float] = []
    best_loss = float("inf")
    no_improve_count = 0

    for epoch in range(config.epochs):
        lr = _scheduled_lr(config.lr, epoch, config.epochs, schedule, warmup_fraction)

        # Shuffle and process mini-batches; accumulate mean epoch loss.
        indices = rng.permutation(n)
        epoch_losses: list[float] = []
        for start in range(0, n, effective_batch):
            batch_idx = indices[start : start + effective_batch]
            state, batch_loss = train_step(
                state, x_pos[batch_idx], x_neg[batch_idx], lr=lr
            )
            epoch_losses.append(batch_loss)

        loss = float(np.mean(epoch_losses))
        losses.append(loss)

        for cb in cbs:
            if hasattr(cb, 'on_epoch_end'):
                cb.on_epoch_end(epoch=epoch, loss=loss, lr=lr)

        if patience > 0:
            if loss < best_loss - min_delta:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.debug(
                        "Early stopping at epoch %d/%d (no improvement for %d epochs)",
                        epoch + 1,
                        config.epochs,
                        patience,
                    )
                    break
        else:
            best_loss = min(best_loss, loss)

    final_loss = losses[-1] if losses else float("nan")
    for cb in cbs:
        if hasattr(cb, 'on_train_end'):
            cb.on_train_end(epochs_run=len(losses), final_loss=final_loss)

    logger.debug(
        "Layer trained: %d/%d epochs, final_loss=%.4f",
        len(losses),
        config.epochs,
        final_loss,
    )
    return state, losses
