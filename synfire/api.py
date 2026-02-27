"""SynfirePipeline: unified public API for time series analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import SynfireConfig
from synfire.layers.ff_stack import FFStackState, init_stack, train_stack
from synfire.layers.hebbian import HebbianState, init_hebbian, train_hebbian
from synfire.pipeline.anomaly import AnomalyScaler, anomaly_scores, fit_anomaly_scaler
from synfire.pipeline.cluster import cluster_assign
from synfire.pipeline.representation import get_representation
from synfire.preprocessing.normalization import normalize_windows
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    sliding_windows,
)


class SynfirePipeline:
    """End-to-end pipeline: raw time series -> windows -> FF features -> Hebbian clusters.

    Follows scikit-learn conventions with fit/transform/predict-style methods.

    Example::

        pipeline = SynfirePipeline()
        pipeline.fit(normal_time_series)
        scores = pipeline.anomaly_scores(test_series)
        clusters = pipeline.cluster(test_series)
        representations = pipeline.transform(test_series)
    """

    def __init__(self, config: SynfireConfig | None = None) -> None:
        self.config = config or SynfireConfig()
        self._stack: FFStackState | None = None
        self._hebbian: HebbianState | None = None
        self._anomaly_scaler: AnomalyScaler | None = None
        self._fitted = False

    def _prepare_pairs(
        self, series: NDArray, rng: np.random.Generator
    ) -> tuple[NDArray, NDArray]:
        """Convert raw series to positive/negative FF training pairs."""
        windows = sliding_windows(series, self.config.window)
        windows = normalize_windows(windows, self.config.norm)

        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)

        n = len(pos_l)
        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[:n], neg_r[:n]], axis=1)
        return x_pos, x_neg

    def _prepare_test_pairs(self, series: NDArray) -> NDArray:
        """Convert raw series to consecutive pairs for inference."""
        windows = sliding_windows(series, self.config.window)
        windows = normalize_windows(windows, self.config.norm)
        left, right = make_consecutive_pairs(windows)
        return np.concatenate([left, right], axis=1)

    def fit(self, series: NDArray) -> SynfirePipeline:
        """Fit the pipeline on normal (training) time series data.

        Args:
            series: 1D array of shape (T,) or 2D of shape (T, C).

        Returns:
            self, for method chaining.
        """
        rng = np.random.default_rng(self.config.ff_stack.seed)

        x_pos, x_neg = self._prepare_pairs(series, rng)
        input_dim = x_pos.shape[1]

        # Train FF stack
        stack = init_stack(input_dim, self.config.ff_stack)
        self._stack, _ = train_stack(stack, x_pos, x_neg)

        # Extract representations for Hebbian training
        representations = get_representation(self._stack, x_pos)

        # Train Hebbian layer
        hebbian = init_hebbian(representations, self.config.hebbian)
        self._hebbian = train_hebbian(hebbian, representations)

        # Fit anomaly scaler on training data for deterministic scoring
        self._anomaly_scaler = fit_anomaly_scaler(
            self._stack,
            self._hebbian,
            x_pos,
            self.config.anomaly,
            self.config.ff_stack.threshold,
        )

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

    def anomaly_scores(self, series: NDArray) -> NDArray:
        """Compute anomaly scores for a test time series.

        Args:
            series: 1D or 2D time series array.

        Returns:
            Scores of shape (N-1,) where N is the number of windows.
            Higher values indicate more anomalous regions.
        """
        self._check_fitted()
        if self._stack is None or self._hebbian is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack or hebbian.")
        test_pairs = self._prepare_test_pairs(series)
        return anomaly_scores(
            self._stack,
            self._hebbian,
            test_pairs,
            self.config.anomaly,
            self.config.ff_stack.threshold,
            scaler=self._anomaly_scaler,
        )

    def cluster(self, series: NDArray) -> NDArray:
        """Assign cluster labels to windows of a time series.

        Args:
            series: 1D or 2D time series array.

        Returns:
            Cluster indices of shape (N-1,).
        """
        self._check_fitted()
        if self._stack is None or self._hebbian is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack or hebbian.")
        test_pairs = self._prepare_test_pairs(series)
        representations = get_representation(self._stack, test_pairs)
        return cluster_assign(self._hebbian, representations)

    def transform(self, series: NDArray) -> NDArray:
        """Extract learned representations for windows of a time series.

        Args:
            series: 1D or 2D time series array.

        Returns:
            Representations of shape (N-1, repr_dim).
        """
        self._check_fitted()
        if self._stack is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack.")
        test_pairs = self._prepare_test_pairs(series)
        return get_representation(self._stack, test_pairs)

    def save(self, path: str | Path) -> None:
        """Save the fitted pipeline to an .npz file.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the pipeline is not fitted.
        """
        from synfire.persistence import save_pipeline

        save_pipeline(self, path)

    @classmethod
    def load(cls, path: str | Path) -> SynfirePipeline:
        """Load a fitted pipeline from an .npz file.

        Args:
            path: Path to the .npz file.

        Returns:
            A fitted SynfirePipeline instance.
        """
        from synfire.persistence import load_pipeline

        return load_pipeline(path)
