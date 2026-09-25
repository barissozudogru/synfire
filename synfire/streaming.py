"""Streaming anomaly scorer for online/real-time use cases."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from synfire.api import SynfirePipeline


class StreamingScorer:
    """Score a time series one point at a time without recomputing history.

    Maintains an internal buffer of ``window_size + 1`` raw values. Once the
    buffer is full, each new data point triggers:

    1. Extraction of two consecutive windows (oldest and newest).
    2. Normalization via the fitted pipeline's ``NormConfig``.
    3. Anomaly scoring via the fitted pipeline's stack, Hebbian layer, and
       anomaly scaler.

    Until the buffer is full, ``score_point`` returns ``None``.

    Example::

        scorer = StreamingScorer.from_pipeline(pipeline)
        for value in stream:
            score = scorer.score_point(value)
            if score is not None and score > threshold:
                alert(score)
    """

    def __init__(
        self,
        pipeline: SynfirePipeline,
    ) -> None:
        if not pipeline._fitted:
            raise RuntimeError("Pipeline must be fitted before creating a StreamingScorer.")
        self._pipeline = pipeline
        self._window_size = pipeline.config.window.window_size
        # Buffer holds window_size + 1 raw scalar/vector values so we can form
        # two consecutive windows: buffer[:-1] and buffer[1:].
        self._buffer: deque = deque(maxlen=self._window_size + 1)
        self._prev_label: int | None = None

    @classmethod
    def from_pipeline(cls, pipeline: SynfirePipeline) -> StreamingScorer:
        """Create a StreamingScorer from a fitted SynfirePipeline.

        Args:
            pipeline: A fitted pipeline (``pipeline.fit()`` must have been called).

        Returns:
            StreamingScorer ready to ingest data points.

        Raises:
            RuntimeError: If the pipeline is not fitted.
        """
        if not pipeline._fitted:
            raise RuntimeError(
                "Pipeline must be fitted before creating a StreamingScorer. "
                "Call pipeline.fit() first."
            )
        return cls(pipeline)

    def score_point(self, value: float | NDArray) -> float | None:
        """Ingest one data point and return an anomaly score when ready.

        Args:
            value: A scalar (univariate) or 1D array of shape (C,) (multivariate).
                All calls after the first must supply a value with the same
                number of channels ``C`` as the first call; a mismatch raises
                ``ValueError`` immediately rather than silently corrupting the
                window buffer.

        Returns:
            Anomaly score (float) once the buffer contains ``window_size + 1``
            points; ``None`` while the buffer is still filling.

        Raises:
            ValueError: If ``value`` has more than one dimension, if it
                contains non-finite values (NaN or Inf), or if its channel
                count differs from previously ingested points.
        """
        orig_arr = np.asarray(value, dtype=np.float64)

        if orig_arr.ndim > 1:
            raise ValueError(
                f"score_point expects a scalar or 1-D array, got shape {orig_arr.shape}"
            )

        arr = orig_arr.ravel()

        # A single NaN or Inf would silently turn every score built from the
        # buffer into NaN until it leaves again, so reject it up front like
        # the batch API does.
        if not np.isfinite(arr).all():
            raise ValueError("score_point: value contains non-finite values (NaN or Inf)")

        # Validate channel count consistency once the buffer has at least one entry.
        if self._buffer:
            expected_channels = self._buffer[0].shape[0]
            if arr.shape[0] != expected_channels:
                raise ValueError(
                    f"Channel count mismatch: expected {expected_channels} channel(s) "
                    f"based on previous inputs, got {arr.shape[0]}. "
                    "All calls to score_point must supply the same number of channels."
                )

        self._buffer.append(arr)

        if len(self._buffer) < self._window_size + 1:
            return None

        # Build the two consecutive windows from the buffer
        buf = np.stack(list(self._buffer))  # (window_size + 1, C)
        # Flatten each window: (window_size * C,)
        w = self._window_size
        left_raw = buf[:w].ravel()
        right_raw = buf[1:].ravel()

        # Normalize using the pipeline's norm config (window-level z-score/minmax)
        left_norm = self._normalize_window(left_raw)
        right_norm = self._normalize_window(right_raw)

        # Build the pair and score it
        pair = np.concatenate([left_norm, right_norm])[np.newaxis, :]  # (1, 2*D)

        if self._pipeline._stack is None or self._pipeline._hebbian is None:
            raise RuntimeError("Pipeline state is corrupted: missing stack or hebbian.")

        from synfire.pipeline.anomaly import _normalize_fixed, anomaly_scores

        scores = anomaly_scores(
            self._pipeline._stack,
            self._pipeline._hebbian,
            pair,
            self._pipeline.config.anomaly,
            self._pipeline._effective_threshold,
            scaler=self._pipeline._anomaly_scaler,
        )
        score = float(scores[0])

        anomaly_cfg = self._pipeline.config.anomaly
        scaler = self._pipeline._anomaly_scaler
        if anomaly_cfg.use_transition and scaler is not None:
            from synfire.layers.ff_stack import forward_stack
            from synfire.layers.hebbian import assign

            acts = forward_stack(self._pipeline._stack, pair)
            cur_label = int(assign(self._pipeline._hebbian, acts[-1])[0])
            if self._prev_label is not None:
                surprise = -np.log(scaler.trans_prob[self._prev_label, cur_label] + 1e-12)
                norm_surprise = float(
                    _normalize_fixed(
                        np.array([surprise]), scaler.surprise_min, scaler.surprise_range
                    )[0]
                )
                score += anomaly_cfg.weight_transition * norm_surprise
            self._prev_label = cur_label

        return score

    def _normalize_window(self, window: NDArray) -> NDArray:
        """Apply the pipeline's normalization to a single flattened window."""
        from synfire.preprocessing.normalization import normalize_windows

        # normalize_windows expects shape (N, D); wrap and unwrap.
        normed = normalize_windows(window[np.newaxis, :], self._pipeline.config.norm)
        return normed[0]

    @property
    def buffer_fullness(self) -> int:
        """Current number of buffered data points (max = window_size + 1)."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True once the buffer is full and scores can be produced."""
        return len(self._buffer) == self._window_size + 1

    def reset(self) -> None:
        """Clear the internal buffer and transition state."""
        self._buffer.clear()
        self._prev_label = None
