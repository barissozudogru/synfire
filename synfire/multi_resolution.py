"""Multi-resolution Forward-Forward pipeline.

Runs independent SynfirePipeline instances at multiple window sizes and
combines their anomaly scores via weighted averaging or max pooling.
Scores are resampled to a common length (the shortest output) before combining.

Example::

    from synfire.multi_resolution import MultiResolutionPipeline
    pipeline = MultiResolutionPipeline(window_sizes=[8, 16, 32])
    pipeline.fit(normal_series)
    scores = pipeline.anomaly_scores(test_series)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from synfire.api import SynfirePipeline
from synfire.core.config import (
    AnomalyConfig,
    FFStackConfig,
    HebbianConfig,
    NormConfig,
    SynfireConfig,
    WindowConfig,
)

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_SIZES = (8, 16, 32, 64)


class MultiResolutionPipeline:
    """Anomaly detection pipeline operating at multiple temporal resolutions.

    Fits an independent :class:`~synfire.api.SynfirePipeline` for each window
    size. At inference time the per-resolution scores are resampled to the
    shortest common length and combined via weighted averaging or max pooling.

    Parameters
    ----------
    window_sizes:
        Sequence of window sizes to run. Defaults to (8, 16, 32, 64).
    base_config:
        Template :class:`~synfire.core.config.SynfireConfig` whose ``window``
        field is overridden per resolution. All other settings are shared
        across resolutions. A lightweight default config is used when None.
    weights:
        Optional per-resolution weights for the weighted average combination.
        Must have the same length as ``window_sizes``. When None, all
        resolutions are weighted equally.
    combination:
        How to combine per-resolution scores. ``"mean"`` uses (weighted)
        averaging; ``"max"`` uses element-wise maximum pooling (ignores
        ``weights``).
    """

    def __init__(
        self,
        window_sizes: tuple[int, ...] | list[int] = _DEFAULT_WINDOW_SIZES,
        base_config: SynfireConfig | None = None,
        weights: list[float] | None = None,
        combination: str = "mean",
    ) -> None:
        if combination not in ("mean", "max"):
            raise ValueError(f"combination must be 'mean' or 'max', got {combination!r}")

        self.window_sizes = list(window_sizes)
        if not self.window_sizes:
            raise ValueError("window_sizes must be non-empty")

        if weights is not None:
            if len(weights) != len(self.window_sizes):
                raise ValueError(
                    f"weights length {len(weights)} must match "
                    f"window_sizes length {len(self.window_sizes)}"
                )
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self._weights = [w / total for w in weights]
        else:
            n = len(self.window_sizes)
            self._weights = [1.0 / n] * n

        self.combination = combination
        self._base_config = base_config or self._default_config()
        self._pipelines: list[SynfirePipeline] = []
        self._fitted = False

    @staticmethod
    def _default_config() -> SynfireConfig:
        """Lightweight default config suitable for multi-resolution use."""
        return SynfireConfig(
            norm=NormConfig(method="zscore"),
            ff_stack=FFStackConfig(
                layer_dims=(32, 16),
                lr=0.05,
                threshold=2.0,
                epochs_per_layer=30,
            ),
            hebbian=HebbianConfig(n_prototypes=8, epochs=20),
            anomaly=AnomalyConfig(),
        )

    def _config_for_window(self, window_size: int) -> SynfireConfig:
        """Create a config with the given window_size, inheriting all other settings."""
        return self._base_config.replace(
            window=WindowConfig(
                window_size=window_size,
                stride=self._base_config.window.stride,
            )
        )

    def fit(self, series: NDArray) -> MultiResolutionPipeline:
        """Fit one pipeline per window size on the provided time series.

        Args:
            series: 1D or 2D normal (training) time series.

        Returns:
            self, for method chaining.
        """
        self._pipelines = []
        for ws in self.window_sizes:
            logger.info("MultiResolutionPipeline: fitting window_size=%d", ws)
            config = self._config_for_window(ws)
            pipeline = SynfirePipeline(config)
            pipeline.fit(series)
            self._pipelines.append(pipeline)
        self._fitted = True
        logger.info(
            "MultiResolutionPipeline fit complete: %d resolutions", len(self.window_sizes)
        )
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("MultiResolutionPipeline not fitted. Call fit() first.")

    def anomaly_scores(self, series: NDArray) -> NDArray:
        """Compute combined anomaly scores across all resolutions.

        Per-resolution scores are resampled to the minimum output length using
        nearest-neighbor interpolation, then combined via the configured method.

        Args:
            series: 1D or 2D time series array.

        Returns:
            Combined anomaly scores of shape (min_output_length,).
        """
        self._check_fitted()

        per_resolution: list[NDArray] = []
        for pipeline in self._pipelines:
            scores = pipeline.anomaly_scores(series)
            per_resolution.append(scores)

        return self._combine(per_resolution)

    def score_decomposed_per_resolution(self, series: NDArray) -> list[NDArray]:
        """Return per-resolution anomaly score arrays (not combined).

        Args:
            series: 1D or 2D time series array.

        Returns:
            List of score arrays, one per window size. Lengths may differ.
        """
        self._check_fitted()
        return [p.anomaly_scores(series) for p in self._pipelines]

    def _combine(self, per_resolution: list[NDArray]) -> NDArray:
        """Resample all score arrays to a common length and combine."""
        min_len = min(len(s) for s in per_resolution)
        if min_len == 0:
            return np.zeros(0)

        resampled = []
        for scores in per_resolution:
            if len(scores) == min_len:
                resampled.append(scores)
            else:
                # Nearest-neighbor resample
                indices = np.round(
                    np.linspace(0, len(scores) - 1, min_len)
                ).astype(int)
                resampled.append(scores[indices])

        stacked = np.stack(resampled, axis=0)  # (n_resolutions, min_len)

        if self.combination == "max":
            return stacked.max(axis=0)

        # Weighted mean
        w = np.array(self._weights, dtype=np.float64)[:, np.newaxis]
        return (stacked * w).sum(axis=0)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"MultiResolutionPipeline({status}, window_sizes={self.window_sizes}, "
            f"combination={self.combination!r})"
        )
