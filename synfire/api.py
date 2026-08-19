"""SynfirePipeline: unified public API for time series analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import SynfireConfig
from synfire.layers.ff_layer import goodness
from synfire.layers.ff_stack import FFStackState, forward_stack, init_stack, train_stack
from synfire.layers.hebbian import HebbianState, init_hebbian, train_hebbian
from synfire.pipeline.anomaly import (
    AnomalyScaler,
    DecomposedAnomalyScore,
    anomaly_scores,
    anomaly_scores_decomposed,
    fit_anomaly_scaler,
)
from synfire.pipeline.cluster import cluster_assign
from synfire.pipeline.representation import get_representation
from synfire.preprocessing.normalization import normalize_windows
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    sliding_windows,
)

logger = logging.getLogger(__name__)


def _validate_series(series: NDArray, window_size: int, method: str) -> NDArray:
    """Validate time series input for pipeline methods."""
    try:
        arr = np.asarray(series)
    except Exception as e:
        raise ValueError(f"{method}: input could not be converted to numpy array: {e}") from e
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{method}: series must have numeric dtype, got dtype={arr.dtype}")
    if arr.ndim not in (1, 2):
        raise ValueError(f"{method}: series must be 1D or 2D, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{method}: series must be non-empty")
    length = arr.shape[0]
    if length < window_size:
        raise ValueError(
            f"{method}: series length {length} is shorter than window_size={window_size}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{method}: series contains non-finite values (NaN or Inf)")
    return arr



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
        # Per-layer loss histories from the most recent fit() call.
        # Shape: list of lists, one inner list per FF stack layer.
        self.training_history: list[list[float]] = []
        # Effective goodness threshold used for scoring (may differ from config
        # when adaptive_threshold=True).
        self._effective_threshold: float = self.config.ff_stack.threshold

    def _prepare_pairs(
        self, series: NDArray, rng: np.random.Generator
    ) -> tuple[NDArray, NDArray]:
        """Convert raw series to positive/negative FF training pairs."""
        windows = sliding_windows(series, self.config.window)
        windows = normalize_windows(windows, self.config.norm)

        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)

        n = len(pos_l)
        if n < 10:
            raise ValueError(
                f"Too few training pairs generated: {n} (minimum 10 required). "
                "Provide a longer time series or reduce window_size/stride to "
                "generate more sliding windows."
            )

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
        series = _validate_series(series, self.config.window.window_size, "fit")
        logger.info(
            "Fitting SynfirePipeline: series shape=%s, window_size=%d, layer_dims=%s",
            series.shape,
            self.config.window.window_size,
            self.config.ff_stack.layer_dims,
        )
        rng = np.random.default_rng(self.config.ff_stack.seed)

        x_pos, x_neg = self._prepare_pairs(series, rng)
        input_dim = x_pos.shape[1]
        logger.info("Prepared %d positive pairs, input_dim=%d", len(x_pos), input_dim)

        # Train FF stack
        stack = init_stack(input_dim, self.config.ff_stack)
        self._stack, all_losses = train_stack(stack, x_pos, x_neg)
        self.training_history = all_losses

        # Adaptive threshold: recalibrate to mean goodness of positive training data.
        # This removes the need to hand-tune the threshold for new datasets.
        if self.config.adaptive_threshold:
            train_activations = forward_stack(self._stack, x_pos)
            pos_goodness = goodness(train_activations[-1])
            self._effective_threshold = float(np.mean(pos_goodness))
            logger.info(
                "Adaptive threshold: config=%.3f -> calibrated=%.3f (mean pos goodness)",
                self.config.ff_stack.threshold,
                self._effective_threshold,
            )
        else:
            self._effective_threshold = self.config.ff_stack.threshold

        # Extract representations for Hebbian training
        representations = get_representation(self._stack, x_pos)

        # Train Hebbian layer
        logger.info(
            "Training Hebbian layer: n_prototypes=%d, epochs=%d",
            self.config.hebbian.n_prototypes,
            self.config.hebbian.epochs,
        )
        hebbian = init_hebbian(representations, self.config.hebbian)
        self._hebbian = train_hebbian(hebbian, representations)

        # Fit anomaly scaler on training data for deterministic scoring
        self._anomaly_scaler = fit_anomaly_scaler(
            self._stack,
            self._hebbian,
            x_pos,
            self.config.anomaly,
            self._effective_threshold,
        )

        self._fitted = True
        logger.info("Pipeline fit complete")
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

    def score_index_to_sample(self, score_index: int) -> int:
        """Map an index in :meth:`anomaly_scores` back to a sample index.

        Scores are offset from the input by the window geometry, so an anomaly
        cannot be located in time without this mapping.

        Args:
            score_index: Position within the array returned by
                :meth:`anomaly_scores`.

        Returns:
            Index into the original series of the first sample of the window
            the score describes.
        """
        w = self.config.window
        return int(score_index) * w.stride + w.stride

    def score_window_bounds(self, score_index: int) -> tuple[int, int]:
        """Half-open sample range ``[start, end)`` covered by a score.

        Args:
            score_index: Position within the array returned by
                :meth:`anomaly_scores`.

        Returns:
            Start and end sample indices of the window the score describes.
        """
        start = self.score_index_to_sample(score_index)
        return start, start + self.config.window.window_size

    def anomaly_scores(self, series: NDArray) -> NDArray:
        """Compute anomaly scores for a test time series.

        Scores are computed over sliding windows, so the output is shorter than
        the input and offset from it. For ``window_size=w`` and ``stride=s``
        there are ``N = (len(series) - w) // s + 1`` windows and ``N - 1``
        scores, because each score compares consecutive windows.

        ``scores[i]`` describes the transition into the window starting at
        ``i * s + s``, so the sample it points at is
        ``score_index_to_sample(i)``. Use that rather than indexing the input
        with a score index directly, which is off by the window offset.

        Args:
            series: 1D or 2D time series array.

        Returns:
            Scores of shape (N-1,). Higher values indicate more anomalous
            regions.

        Example:
            >>> scores = pipeline.anomaly_scores(series)   # len 275 for 300 in
            >>> worst = int(scores.argmax())
            >>> sample = pipeline.score_index_to_sample(worst)
        """
        series = _validate_series(series, self.config.window.window_size, "anomaly_scores")
        self._check_fitted()
        if self._stack is None or self._hebbian is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack or hebbian.")
        test_pairs = self._prepare_test_pairs(series)
        return anomaly_scores(
            self._stack,
            self._hebbian,
            test_pairs,
            self.config.anomaly,
            self._effective_threshold,
            scaler=self._anomaly_scaler,
        )

    def score_decomposed(self, series: NDArray) -> DecomposedAnomalyScore:
        """Compute decomposed anomaly scores with per-component breakdown.

        Returns each scoring component separately (goodness_deficit,
        prototype_distance, transition_surprise) alongside the combined score.
        Components disabled in config are returned as None.

        Args:
            series: 1D or 2D time series array.

        Returns:
            DecomposedAnomalyScore dataclass with individual and combined scores.
        """
        series = _validate_series(series, self.config.window.window_size, "score_decomposed")
        self._check_fitted()
        if self._stack is None or self._hebbian is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack or hebbian.")
        test_pairs = self._prepare_test_pairs(series)
        return anomaly_scores_decomposed(
            self._stack,
            self._hebbian,
            test_pairs,
            self.config.anomaly,
            self._effective_threshold,
            scaler=self._anomaly_scaler,
        )

    def cluster(self, series: NDArray) -> NDArray:
        """Assign cluster labels to windows of a time series.

        Args:
            series: 1D or 2D time series array.

        Returns:
            Cluster indices of shape (N-1,).
        """
        series = _validate_series(series, self.config.window.window_size, "cluster")
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
        series = _validate_series(series, self.config.window.window_size, "transform")
        self._check_fitted()
        if self._stack is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack.")
        test_pairs = self._prepare_test_pairs(series)
        return get_representation(self._stack, test_pairs)

    def fit_transform(self, series: NDArray) -> NDArray:
        """Fit the pipeline and return learned representations.

        Args:
            series: 1D array of shape (T,) or 2D of shape (T, C).

        Returns:
            Representations of shape (N-1, repr_dim).
        """
        return self.fit(series).transform(series)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        ws = self.config.window.window_size
        dims = self.config.ff_stack.layer_dims
        n_proto = self.config.hebbian.n_prototypes
        return (
            f"SynfirePipeline({status}, window_size={ws}, "
            f"layer_dims={dims}, n_prototypes={n_proto})"
        )

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
