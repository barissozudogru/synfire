"""Per-window normalization for time series windows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import NormConfig


def normalize_windows(windows: NDArray, config: NormConfig | None = None) -> NDArray:
    """Normalize each window independently.

    Args:
        windows: Array of shape (N, D).
        config: Normalization config. Uses defaults if None.

    Returns:
        Normalized array of same shape.
    """
    if config is None:
        config = NormConfig()

    if config.method == "zscore":
        mean = np.mean(windows, axis=1, keepdims=True)
        std = np.std(windows, axis=1, keepdims=True)
        return (windows - mean) / (std + config.eps)
    elif config.method == "minmax":
        wmin = np.min(windows, axis=1, keepdims=True)
        wmax = np.max(windows, axis=1, keepdims=True)
        return (windows - wmin) / (wmax - wmin + config.eps)
    else:
        raise ValueError(f"Unknown normalization method: {config.method}")
