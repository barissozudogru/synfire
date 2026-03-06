"""Visualization utilities for synfire training and anomaly scoring results.

All functions require matplotlib. Import errors are deferred so that synfire
can be used without matplotlib installed (it is not a hard dependency).

Typical usage::

    from synfire.visualization import plot_training_loss, plot_anomaly_scores
    plot_training_loss(pipeline.training_history)
    plot_anomaly_scores(scores, threshold=0.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from synfire.pipeline.anomaly import DecomposedAnomalyScore


def _require_matplotlib():
    """Import and return (matplotlib, pyplot) or raise ImportError with guidance."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for synfire visualization. "
            "Install it with: pip install matplotlib"
        ) from exc


def plot_training_loss(
    history: list[list[float]],
    *,
    ax=None,
    title: str = "Training Loss per Layer",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> object:
    """Plot loss curves for each layer in the FF stack.

    Args:
        history: Per-layer loss lists as returned by ``pipeline.training_history``.
            Shape: list of L lists, each containing per-epoch loss values.
        ax: Optional matplotlib Axes. A new figure is created when None.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The matplotlib Axes object.
    """
    _, plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for i, layer_losses in enumerate(history):
        ax.plot(layer_losses, label=f"Layer {i + 1}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if history:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_goodness_distribution(
    pos_goodness: NDArray,
    neg_goodness: NDArray,
    *,
    ax=None,
    bins: int = 50,
    threshold: float | None = None,
    title: str = "Goodness Distribution",
    xlabel: str = "Goodness",
    ylabel: str = "Density",
) -> object:
    """Histogram overlay of positive and negative goodness values.

    Args:
        pos_goodness: Goodness scores for positive (normal) samples, shape (N,).
        neg_goodness: Goodness scores for negative samples, shape (N,).
        ax: Optional matplotlib Axes. A new figure is created when None.
        bins: Number of histogram bins.
        threshold: If provided, draws a vertical line at the threshold value.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The matplotlib Axes object.
    """
    _, plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    common_kwargs = dict(bins=bins, density=True, alpha=0.6)
    ax.hist(pos_goodness, label="Positive", color="steelblue", **common_kwargs)
    ax.hist(neg_goodness, label="Negative", color="salmon", **common_kwargs)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={threshold}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_anomaly_scores(
    scores: NDArray,
    *,
    labels: NDArray | None = None,
    threshold: float | None = None,
    ax=None,
    title: str = "Anomaly Scores",
    xlabel: str = "Time Step",
    ylabel: str = "Score",
) -> object:
    """Time series plot of anomaly scores with optional threshold and ground-truth overlay.

    Args:
        scores: Anomaly scores of shape (T,).
        labels: Optional binary ground-truth anomaly labels of shape (T,).
            Anomalous regions (label == 1) are shaded red.
        threshold: If provided, draws a horizontal threshold line.
        ax: Optional matplotlib Axes. A new figure is created when None.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The matplotlib Axes object.
    """
    _, plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    t = np.arange(len(scores))
    ax.plot(t, scores, color="steelblue", linewidth=0.8, label="Score")

    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold={threshold:.3f}")

    if labels is not None:
        labels = np.asarray(labels, dtype=bool)
        # Shade anomalous regions
        in_anomaly = False
        start = 0
        for i, lbl in enumerate(labels):
            if lbl and not in_anomaly:
                start = i
                in_anomaly = True
            elif not lbl and in_anomaly:
                ax.axvspan(start, i, alpha=0.25, color="red", label="_nolegend_")
                in_anomaly = False
        if in_anomaly:
            ax.axvspan(start, len(labels), alpha=0.25, color="red", label="_nolegend_")
        # Add a single legend entry for anomaly regions
        from matplotlib.patches import Patch
        ax.legend(handles=[
            ax.get_lines()[0],
            *(ax.get_lines()[1:] if threshold is not None else []),
            Patch(facecolor="red", alpha=0.25, label="Anomaly"),
        ])
    else:
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def plot_score_decomposition(
    decomposed: DecomposedAnomalyScore,
    *,
    ax=None,
    title: str = "Anomaly Score Decomposition",
    xlabel: str = "Time Step",
    ylabel: str = "Score",
) -> object:
    """Stacked area chart showing each anomaly score component over time.

    Args:
        decomposed: DecomposedAnomalyScore from ``pipeline.score_decomposed()``.
        ax: Optional matplotlib Axes. A new figure is created when None.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The matplotlib Axes object.
    """
    _, plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    components: list[tuple[str, NDArray]] = []
    n = len(decomposed.combined)

    if decomposed.goodness_deficit is not None:
        components.append(("Goodness Deficit", decomposed.goodness_deficit))
    if decomposed.prototype_distance is not None:
        components.append(("Prototype Distance", decomposed.prototype_distance))
    if decomposed.transition_surprise is not None:
        components.append(("Transition Surprise", decomposed.transition_surprise))

    t = np.arange(n)
    colors = ["steelblue", "darkorange", "seagreen"]

    if components:
        labels = [c[0] for c in components]
        arrays = [c[1] for c in components]
        ax.stackplot(t, *arrays, labels=labels, colors=colors[:len(components)], alpha=0.75)
        ax.legend(loc="upper left")
    else:
        ax.plot(t, decomposed.combined, color="steelblue", label="Combined")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax
