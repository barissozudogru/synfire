"""NumPy backend abstraction for synfire computations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class NumPyBackend:
    """Thin wrapper around NumPy ops used throughout synfire.

    Provides a single point of replacement if we later swap in JAX.
    """

    def relu(self, x: NDArray) -> NDArray:
        return np.maximum(x, 0.0)

    def sigmoid(self, x: NDArray) -> NDArray:
        # Numerically stable sigmoid
        pos = x >= 0
        z = np.empty_like(x)
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        z[~pos] = exp_x / (1.0 + exp_x)
        return z

    def matmul(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def norm(self, x: NDArray, axis: int = -1, keepdims: bool = True) -> NDArray:
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    def mean(self, x: NDArray, axis: int | None = None) -> NDArray:
        return np.mean(x, axis=axis)

    def sum(self, x: NDArray, axis: int | None = None) -> NDArray:
        return np.sum(x, axis=axis)

    def log(self, x: NDArray) -> NDArray:
        return np.log(np.clip(x, 1e-12, None))

    def clip(self, x: NDArray, lo: float, hi: float) -> NDArray:
        return np.clip(x, lo, hi)

    def zeros(self, shape: tuple[int, ...], dtype: type = np.float64) -> NDArray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: type = np.float64) -> NDArray:
        return np.ones(shape, dtype=dtype)


backend = NumPyBackend()
