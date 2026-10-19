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

    if windows.ndim != 2:
        raise ValueError(f"windows must be 2D, got ndim={windows.ndim}")

    if config.method == "zscore":
        mean = np.mean(windows, axis=1, keepdims=True)
        std = np.std(windows, axis=1, keepdims=True)
        numerator = windows - mean
        denominator = std + config.eps
    elif config.method == "minmax":
        wmin = np.min(windows, axis=1, keepdims=True)
        wmax = np.max(windows, axis=1, keepdims=True)
        numerator = windows - wmin
        denominator = wmax - wmin + config.eps
    else:
        raise ValueError(f"Unknown normalization method: {config.method}")

    # A constant window has no scale. Treat it as all zeros, including when
    # eps=0 is explicitly requested, rather than returning NaN from 0 / 0.
    denominator = np.where(denominator == 0, 1, denominator)
    return numerator / denominator
