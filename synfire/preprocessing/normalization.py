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

    # Promote low-precision inputs while calculating statistics so squaring
    # large float32/float16 values cannot overflow before normalization.
    calculation_dtype = np.result_type(windows.dtype, np.float64)
    calculation_windows = windows.astype(calculation_dtype, copy=False)

    if config.method == "zscore":
        mean = np.mean(calculation_windows, axis=1, keepdims=True)
        std = np.std(calculation_windows, axis=1, keepdims=True)
        numerator = calculation_windows - mean
        denominator = std + config.eps
    elif config.method == "minmax":
        wmin = np.min(calculation_windows, axis=1, keepdims=True)
        wmax = np.max(calculation_windows, axis=1, keepdims=True)
        numerator = calculation_windows - wmin
        denominator = wmax - wmin + config.eps
    else:
        raise ValueError(f"Unknown normalization method: {config.method}")

    # A constant window has no scale. Treat it as all zeros, including when
    # eps=0 is explicitly requested, rather than returning NaN from 0 / 0.
    denominator = np.where(denominator == 0, 1, denominator)
    normalized = numerator / denominator
    if np.issubdtype(windows.dtype, np.floating):
        return normalized.astype(windows.dtype, copy=False)
    return normalized
