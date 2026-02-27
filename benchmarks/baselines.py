"""Baseline anomaly detectors for comparison.

ZScoreBaseline is pure NumPy. WindowedIsolationForest and WindowedLOF
require scikit-learn (optional dependency).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaselineDetector(ABC):
    """Common interface for baseline anomaly detectors."""

    @abstractmethod
    def fit(self, series: NDArray) -> None:
        """Fit the detector on a normal time series."""

    @abstractmethod
    def anomaly_scores(self, series: NDArray) -> NDArray:
        """Compute anomaly scores. Higher = more anomalous."""


class ZScoreBaseline(BaselineDetector):
    """Rolling z-score baseline (pure NumPy).

    Computes z-score of each point relative to a rolling window of
    historical values. Points far from the local mean are scored higher.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._train_mean: float = 0.0
        self._train_std: float = 1.0

    def fit(self, series: NDArray) -> None:
        self._train_mean = float(series.mean())
        self._train_std = float(series.std()) or 1.0

    def anomaly_scores(self, series: NDArray) -> NDArray:
        n = len(series)
        scores = np.zeros(n)
        w = self.window_size

        for i in range(n):
            start = max(0, i - w)
            window = series[start:i + 1]
            if len(window) < 2:
                scores[i] = abs(series[i] - self._train_mean) / self._train_std
            else:
                local_mean = window.mean()
                local_std = window.std() or self._train_std
                scores[i] = abs(series[i] - local_mean) / local_std

        return scores


class WindowedIsolationForest(BaselineDetector):
    """Isolation Forest applied to sliding windows. Requires scikit-learn."""

    def __init__(self, window_size: int = 20, n_estimators: int = 100, seed: int = 42) -> None:
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.seed = seed
        self._model = None

    def _make_windows(self, series: NDArray) -> NDArray:
        n = len(series) - self.window_size + 1
        indices = np.arange(self.window_size)[np.newaxis, :] + np.arange(n)[:, np.newaxis]
        if series.ndim == 1:
            return series[indices]
        return series[indices].reshape(n, -1)

    def fit(self, series: NDArray) -> None:
        from sklearn.ensemble import IsolationForest

        windows = self._make_windows(series)
        self._model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, contamination="auto"
        )
        self._model.fit(windows)

    def anomaly_scores(self, series: NDArray) -> NDArray:
        windows = self._make_windows(series)
        # score_samples returns negative anomaly scores (lower = more anomalous)
        return -self._model.score_samples(windows)


class WindowedLOF(BaselineDetector):
    """Local Outlier Factor applied to sliding windows. Requires scikit-learn."""

    def __init__(self, window_size: int = 20, n_neighbors: int = 20) -> None:
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self._model = None

    def _make_windows(self, series: NDArray) -> NDArray:
        n = len(series) - self.window_size + 1
        indices = np.arange(self.window_size)[np.newaxis, :] + np.arange(n)[:, np.newaxis]
        if series.ndim == 1:
            return series[indices]
        return series[indices].reshape(n, -1)

    def fit(self, series: NDArray) -> None:
        from sklearn.neighbors import LocalOutlierFactor

        windows = self._make_windows(series)
        self._model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        self._model.fit(windows)

    def anomaly_scores(self, series: NDArray) -> NDArray:
        windows = self._make_windows(series)
        return -self._model.score_samples(windows)
