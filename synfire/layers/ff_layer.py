"""Single Forward-Forward layer with local gradient training."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
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
    # Layer normalization parameters (only used when config.layer_norm=True)
    ln_gain: NDArray | None = None  # (hidden_dim,)
    ln_bias: NDArray | None = None  # (hidden_dim,)
    # Adam optimizer first/second moment estimates (only used when config.optimizer="adam")
    m_W: NDArray | None = None   # first moment for W
    v_W: NDArray | None = None   # second moment for W
    m_b: NDArray | None = None   # first moment for b
    v_b: NDArray | None = None   # second moment for b
    adam_t: int = 0              # Adam step counter


def init_layer(config: FFLayerConfig) -> FFLayerState:
    """Initialize a single FF layer with Xavier initialization."""
    rng = np.random.default_rng(config.seed)
    scale = np.sqrt(2.0 / (config.input_dim + config.hidden_dim))
    W = rng.standard_normal((config.hidden_dim, config.input_dim)) * scale
    b = np.zeros(config.hidden_dim)

    ln_gain = np.ones(config.hidden_dim) if config.layer_norm else None
    ln_bias = np.zeros(config.hidden_dim) if config.layer_norm else None

    m_W = np.zeros_like(W) if config.optimizer == "adam" else None
    v_W = np.zeros_like(W) if config.optimizer == "adam" else None
    m_b = np.zeros_like(b) if config.optimizer == "adam" else None
    v_b = np.zeros_like(b) if config.optimizer == "adam" else None

    return FFLayerState(
        W=W, b=b, config=config,
        ln_gain=ln_gain, ln_bias=ln_bias,
        m_W=m_W, v_W=v_W, m_b=m_b, v_b=v_b,
        adam_t=0,
    )


def _apply_layer_norm(
    pre: NDArray, gain: NDArray, bias: NDArray, eps: float = 1e-8
) -> tuple[NDArray, NDArray, NDArray]:
    """Apply layer normalization: h_norm = (pre - mean) / sqrt(var + eps) * gain + bias.

    Returns:
        (normalized, mean, inv_std) for use in backward pass.
    """
    mean = pre.mean(axis=1, keepdims=True)
    var = pre.var(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    h_norm = (pre - mean) * inv_std
    return h_norm * gain + bias, h_norm, inv_std


def forward(state: FFLayerState, x: NDArray) -> NDArray:
    """Forward pass: h = relu(layernorm(x @ W^T + b)) or relu(x @ W^T + b).

    Args:
        state: Layer state with weights and bias.
        x: Input of shape (batch, input_dim).

    Returns:
        Activations of shape (batch, hidden_dim).
    """
    pre = backend.matmul(x, state.W.T) + state.b
    if state.config.layer_norm and state.ln_gain is not None and state.ln_bias is not None:
        normed, _, _ = _apply_layer_norm(pre, state.ln_gain, state.ln_bias)
        return backend.relu(normed)
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


def _backward_layer_norm(
    d_out: NDArray, h_norm: NDArray, inv_std: NDArray, gain: NDArray
) -> NDArray:
    """Gradient through layer normalization.

    d_out has shape (batch, D). Returns gradient w.r.t. pre-norm input.
    """
    D = d_out.shape[1]
    # Scaled gradient
    d_h_norm = d_out * gain  # (batch, D)
    # Standard LN backward
    d_pre = (1.0 / D) * inv_std * (
        D * d_h_norm
        - d_h_norm.sum(axis=1, keepdims=True)
        - h_norm * (d_h_norm * h_norm).sum(axis=1, keepdims=True)
    )
    return d_pre


def _pairwise_sq_distances(a: NDArray, b: NDArray, chunk_size: int = 512) -> NDArray:
    """Compute pairwise squared L2 distances between rows of *a* and *b*.

    Uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a@b^T to avoid
    materialising the full (Bp, Bn, D) tensor.  When the naive allocation
    would exceed ``_HARD_NEG_ELEMENT_LIMIT`` elements the computation is
    split into row-chunks of at most ``chunk_size`` rows of *a*, keeping
    peak memory proportional to ``chunk_size * Bn`` rather than ``Bp * Bn``.

    Args:
        a: Array of shape (Bp, D).
        b: Array of shape (Bn, D).
        chunk_size: Maximum number of rows of *a* processed at once.

    Returns:
        Distance matrix of shape (Bp, Bn).
    """
    Bp, D = a.shape
    Bn = b.shape[0]

    # If the naive tensor fits comfortably, use the fast matrix path.
    if Bp * Bn * D <= _HARD_NEG_ELEMENT_LIMIT:
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (Bp, Bn, D)
        return np.sum(diff ** 2, axis=2)

    # Chunked computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a @ b^T)
    a_sq = np.sum(a ** 2, axis=1)   # (Bp,)
    b_sq = np.sum(b ** 2, axis=1)   # (Bn,)
    dists = np.empty((Bp, Bn), dtype=a.dtype)
    for start in range(0, Bp, chunk_size):
        end = min(start + chunk_size, Bp)
        # (chunk, Bn)
        cross = a[start:end] @ b.T
        dists[start:end] = a_sq[start:end, np.newaxis] + b_sq[np.newaxis, :] - 2.0 * cross
    # Numerical noise can produce tiny negatives; clamp to zero.
    np.maximum(dists, 0.0, out=dists)
    return dists


# Threshold in total elements (Bp * Bn * D) above which chunked distance
# computation is used to avoid O(N^2 * D) peak memory.
_HARD_NEG_ELEMENT_LIMIT: int = 1 << 27  # 128 M elements ~ 1 GB at float64


def _mine_hard_negatives(
    x_pos: NDArray, x_neg: NDArray, state: FFLayerState, epoch: int, total_epochs: int
) -> NDArray:
    """Select hard negatives closest to positive samples in pre-activation space.

    For "curriculum" strategy, interpolates between random and hard based on
    training progress. For "hard", always returns the closest negatives.

    Args:
        x_pos: Positive samples, shape (batch, input_dim).
        x_neg: Negative candidate pool, shape (batch, input_dim).
        state: Current layer state.
        epoch: Current epoch index (0-based).
        total_epochs: Total training epochs.

    Returns:
        Selected negatives of shape (batch, input_dim).
    """
    strategy = state.config.negative_strategy
    if strategy == "random":
        return x_neg

    # Compute L2 distances in input space (cheap proxy for feature-space hardness).
    # Uses chunked computation when Bp * Bn * D would exceed _HARD_NEG_ELEMENT_LIMIT.
    dists = _pairwise_sq_distances(x_pos, x_neg)  # (Bp, Bn)

    # For curriculum: probability of choosing hard negatives increases with epoch
    if strategy == "curriculum":
        hard_fraction = epoch / max(1, total_epochs - 1)
        rng = np.random.default_rng(epoch)
        use_hard = rng.random(len(x_pos)) < hard_fraction
        hard_indices = np.argmin(dists, axis=1)
        random_indices = rng.integers(0, len(x_neg), size=len(x_pos))
        indices = np.where(use_hard, hard_indices, random_indices)
    else:
        # "hard": always pick closest negative
        indices = np.argmin(dists, axis=1)

    return x_neg[indices]


def train_step(
    state: FFLayerState,
    x_pos: NDArray,
    x_neg: NDArray,
    lr: float | None = None,
    epoch: int = 0,
    total_epochs: int = 1,
) -> tuple[FFLayerState, float]:
    """Single training step: forward both, compute loss, update weights.

    Args:
        state: Current layer state.
        x_pos: Positive input pairs, shape (batch, input_dim).
        x_neg: Negative input pairs, shape (batch, input_dim).
        lr: Learning rate override. Uses state.config.lr when None.
        epoch: Current epoch (for curriculum negative mining).
        total_epochs: Total epochs (for curriculum negative mining).

    Returns:
        (updated_state, loss_value).
    """
    effective_lr = lr if lr is not None else state.config.lr

    # Apply hard negative mining if configured
    if state.config.negative_strategy != "random":
        x_neg = _mine_hard_negatives(x_pos, x_neg, state, epoch, total_epochs)

    use_ln = state.config.layer_norm and state.ln_gain is not None and state.ln_bias is not None

    # Forward positive
    pre_pos = backend.matmul(x_pos, state.W.T) + state.b
    if use_ln:
        pre_pos_normed, h_norm_pos, inv_std_pos = _apply_layer_norm(
            pre_pos, state.ln_gain, state.ln_bias  # type: ignore[arg-type]
        )
        h_pos = backend.relu(pre_pos_normed)
    else:
        h_pos = backend.relu(pre_pos)

    g_pos = goodness(h_pos)

    # Forward negative
    pre_neg = backend.matmul(x_neg, state.W.T) + state.b
    if use_ln:
        pre_neg_normed, h_norm_neg, inv_std_neg = _apply_layer_norm(
            pre_neg, state.ln_gain, state.ln_bias  # type: ignore[arg-type]
        )
        h_neg = backend.relu(pre_neg_normed)
    else:
        h_neg = backend.relu(pre_neg)

    g_neg = goodness(h_neg)

    # Loss
    loss_val, d_g_pos, d_g_neg = compute_loss(g_pos, g_neg, state.config.threshold)

    # Backward through goodness -> relu -> [layernorm] -> linear
    d_h_pos = _backward_goodness(h_pos, d_g_pos)
    if use_ln:
        d_pre_pos_normed = _backward_relu(pre_pos_normed, d_h_pos)  # type: ignore[possibly-undefined]
        d_ln_gain_pos = (d_pre_pos_normed * h_norm_pos).sum(axis=0)  # type: ignore[possibly-undefined]
        d_ln_bias_pos = d_pre_pos_normed.sum(axis=0)
        d_pre_pos = _backward_layer_norm(d_pre_pos_normed, h_norm_pos, inv_std_pos, state.ln_gain)  # type: ignore[arg-type,possibly-undefined]
    else:
        d_pre_pos = _backward_relu(pre_pos, d_h_pos)
        d_ln_gain_pos = None
        d_ln_bias_pos = None

    d_h_neg = _backward_goodness(h_neg, d_g_neg)
    if use_ln:
        d_pre_neg_normed = _backward_relu(pre_neg_normed, d_h_neg)  # type: ignore[possibly-undefined]
        d_ln_gain_neg = (d_pre_neg_normed * h_norm_neg).sum(axis=0)  # type: ignore[possibly-undefined]
        d_ln_bias_neg = d_pre_neg_normed.sum(axis=0)
        d_pre_neg = _backward_layer_norm(d_pre_neg_normed, h_norm_neg, inv_std_neg, state.ln_gain)  # type: ignore[arg-type,possibly-undefined]
    else:
        d_pre_neg = _backward_relu(pre_neg, d_h_neg)
        d_ln_gain_neg = None
        d_ln_bias_neg = None

    # Gradients for W and b
    dW = backend.matmul(d_pre_pos.T, x_pos) + backend.matmul(d_pre_neg.T, x_neg)
    db = np.sum(d_pre_pos, axis=0) + np.sum(d_pre_neg, axis=0)

    # Weight decay (L2): adds wd * W to gradient (not applied to biases)
    wd = state.config.weight_decay
    if wd > 0.0:
        dW = dW + wd * state.W

    # Gradient clipping by global norm
    clip_norm = state.config.grad_clip_norm
    if clip_norm > 0.0:
        grad_norm = math.sqrt(float(np.sum(dW ** 2)) + float(np.sum(db ** 2)))
        if grad_norm > clip_norm:
            scale = clip_norm / (grad_norm + 1e-12)
            dW = dW * scale
            db = db * scale

    # Optimizer step
    if state.config.optimizer == "adam":
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        t = state.adam_t + 1
        m_W = beta1 * state.m_W + (1.0 - beta1) * dW  # type: ignore[operator]
        v_W = beta2 * state.v_W + (1.0 - beta2) * (dW ** 2)  # type: ignore[operator]
        m_b = beta1 * state.m_b + (1.0 - beta1) * db  # type: ignore[operator]
        v_b = beta2 * state.v_b + (1.0 - beta2) * (db ** 2)  # type: ignore[operator]
        # Bias-corrected estimates
        m_W_hat = m_W / (1.0 - beta1 ** t)
        v_W_hat = v_W / (1.0 - beta2 ** t)
        m_b_hat = m_b / (1.0 - beta1 ** t)
        v_b_hat = v_b / (1.0 - beta2 ** t)
        new_W = state.W - effective_lr * m_W_hat / (np.sqrt(v_W_hat) + eps_adam)
        new_b = state.b - effective_lr * m_b_hat / (np.sqrt(v_b_hat) + eps_adam)
    else:
        # SGD
        new_W = state.W - effective_lr * dW
        new_b = state.b - effective_lr * db
        m_W = state.m_W
        v_W = state.v_W
        m_b = state.m_b
        v_b = state.v_b
        t = state.adam_t

    if not np.all(np.isfinite(new_W)) or not np.all(np.isfinite(new_b)):
        raise RuntimeError(
            "Non-finite values detected in layer weights after update "
            "(NaN or Inf). Try reducing the learning rate or clipping gradients."
        )

    # Update layer norm parameters
    new_ln_gain = state.ln_gain
    new_ln_bias = state.ln_bias
    if use_ln and d_ln_gain_pos is not None:
        d_ln_gain = d_ln_gain_pos + d_ln_gain_neg  # type: ignore[operator]
        d_ln_bias = d_ln_bias_pos + d_ln_bias_neg  # type: ignore[operator]
        new_ln_gain = state.ln_gain - effective_lr * d_ln_gain  # type: ignore[operator]
        new_ln_bias = state.ln_bias - effective_lr * d_ln_bias  # type: ignore[operator]

    return FFLayerState(
        W=new_W, b=new_b, config=state.config,
        ln_gain=new_ln_gain, ln_bias=new_ln_bias,
        m_W=m_W, v_W=v_W, m_b=m_b, v_b=v_b,
        adam_t=t,
    ), loss_val


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
                state, x_pos[batch_idx], x_neg[batch_idx], lr=lr,
                epoch=epoch, total_epochs=config.epochs,
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
