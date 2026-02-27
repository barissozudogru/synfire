"""Sliding window extraction and pair construction for time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import WindowConfig


def sliding_windows(series: NDArray, config: WindowConfig | None = None) -> NDArray:
    """Extract sliding windows from a time series.

    Args:
        series: 1D array of shape (T,) or 2D array of shape (T, C).
        config: Window configuration. Uses defaults if None.

    Returns:
        Array of shape (N, window_size) for univariate or (N, window_size * C) for multivariate,
        where N = (T - window_size) // stride + 1.
    """
    if config is None:
        config = WindowConfig()

    if series.ndim == 1:
        series = series[:, np.newaxis]

    T, C = series.shape
    w = config.window_size
    s = config.stride

    n_windows = (T - w) // s + 1
    if n_windows <= 0:
        raise ValueError(
            f"Series length {T} too short for window_size={w} and stride={s}"
        )

    indices = np.arange(w)[np.newaxis, :] + np.arange(n_windows)[:, np.newaxis] * s
    windows = series[indices]  # (N, w, C)
    return windows.reshape(n_windows, w * C)


def make_consecutive_pairs(windows: NDArray) -> tuple[NDArray, NDArray]:
    """Create positive pairs from consecutive windows.

    Returns:
        Tuple of (left, right) arrays, each of shape (N-1, D) where D is the
        flattened window dimension.
    """
    return windows[:-1], windows[1:]


def make_random_pairs(
    windows: NDArray, rng: np.random.Generator, min_gap: int = 5
) -> tuple[NDArray, NDArray]:
    """Create negative pairs by pairing each window with a random non-adjacent one.

    Args:
        windows: Array of shape (N, D).
        rng: NumPy random generator.
        min_gap: Minimum index distance between paired windows.

    Returns:
        Tuple of (left, right) arrays, each of shape (N, D).
    """
    n = len(windows)
    left_indices = np.arange(n)
    right_indices = np.empty(n, dtype=np.intp)

    for i in range(n):
        candidates = np.where(np.abs(np.arange(n) - i) >= min_gap)[0]
        if len(candidates) == 0:
            candidates = np.array([j for j in range(n) if j != i])
        right_indices[i] = rng.choice(candidates)

    return windows[left_indices], windows[right_indices]


def make_shuffled_pairs(
    windows: NDArray, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """Create negative pairs by shuffling the second element.

    Returns:
        Tuple of (left, shuffled_right) arrays.
    """
    perm = rng.permutation(len(windows))
    return windows, windows[perm]
