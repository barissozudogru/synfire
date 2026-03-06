"""Tests for synfire.streaming.StreamingScorer."""

from __future__ import annotations

import numpy as np
import pytest

from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import FFStackConfig, HebbianConfig, WindowConfig
from synfire.streaming import StreamingScorer


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Small fitted pipeline used across streaming tests."""
    rng = np.random.default_rng(42)
    t = np.arange(500, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)

    config = SynfireConfig(
        window=WindowConfig(window_size=10, stride=1),
        ff_stack=FFStackConfig(layer_dims=(16,), lr=0.01, threshold=2.0, epochs_per_layer=5),
        hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=5),
    )
    pipeline = SynfirePipeline(config)
    pipeline.fit(series)
    return pipeline


@pytest.fixture(scope="module")
def unfitted_pipeline():
    """An unfitted pipeline for error-path tests."""
    config = SynfireConfig(
        window=WindowConfig(window_size=10, stride=1),
        ff_stack=FFStackConfig(layer_dims=(16,), lr=0.01, threshold=2.0, epochs_per_layer=5),
        hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=5),
    )
    return SynfirePipeline(config)


class TestUnfittedPipelineRaisesError:
    """Verify that constructing a StreamingScorer from an unfitted pipeline fails."""

    def test_direct_init_raises(self, unfitted_pipeline):
        with pytest.raises(RuntimeError, match="fitted"):
            StreamingScorer(unfitted_pipeline)

    def test_from_pipeline_classmethod_raises(self, unfitted_pipeline):
        with pytest.raises(RuntimeError, match="fitted"):
            StreamingScorer.from_pipeline(unfitted_pipeline)


class TestBufferFilling:
    """Verify that score_point returns None while the buffer is filling."""

    def test_returns_none_before_buffer_full(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        # Need window_size + 1 points before a score is produced
        for i in range(window_size):
            result = scorer.score_point(float(i) * 0.1)
            assert result is None, f"Expected None at step {i}, got {result}"

    def test_returns_float_when_buffer_full(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        result = None
        for i in range(window_size + 1):
            result = scorer.score_point(np.sin(i * 0.1))
        assert result is not None
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_buffer_fullness_tracks_correctly(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        assert scorer.buffer_fullness == 0
        assert not scorer.is_ready

        for i in range(window_size + 1):
            scorer.score_point(float(i))

        assert scorer.buffer_fullness == window_size + 1
        assert scorer.is_ready


class TestScorePoint:
    """Tests for score_point output properties."""

    def test_score_is_finite(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        scores = []
        t = np.arange(window_size + 20, dtype=np.float64)
        series = np.sin(2 * np.pi * t / 50)
        for v in series:
            score = scorer.score_point(v)
            if score is not None:
                scores.append(score)
        assert len(scores) > 0
        assert all(np.isfinite(s) for s in scores)

    def test_consecutive_scores_produced_after_buffer_full(self, fitted_pipeline):
        """After buffer is full, every new point should produce a score."""
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        # Fill buffer
        for i in range(window_size + 1):
            scorer.score_point(float(i) * 0.01)

        # Now every additional point should yield a non-None score
        for i in range(5):
            result = scorer.score_point(float(i) * 0.01)
            assert result is not None

    def test_numpy_array_input_accepted(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        result = None
        for i in range(window_size + 1):
            result = scorer.score_point(np.array(float(i) * 0.1))
        assert result is not None
        assert isinstance(result, float)


class TestReset:
    """Tests for the reset() method."""

    def test_reset_clears_buffer(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        # Fill buffer
        for i in range(window_size + 1):
            scorer.score_point(float(i))
        assert scorer.is_ready

        scorer.reset()

        assert scorer.buffer_fullness == 0
        assert not scorer.is_ready

    def test_after_reset_buffer_fills_again(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size

        # Fill, reset, then refill
        for i in range(window_size + 1):
            scorer.score_point(float(i))
        scorer.reset()

        result = None
        for i in range(window_size + 1):
            result = scorer.score_point(float(i) * 0.1)
        assert result is not None
        assert isinstance(result, float)

    def test_reset_returns_none_on_next_points(self, fitted_pipeline):
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        for i in range(window_size + 1):
            scorer.score_point(float(i))
        scorer.reset()

        # After reset, partial fills should return None
        for i in range(window_size):
            result = scorer.score_point(float(i))
            assert result is None


class TestFromPipelineClassMethod:
    """Tests for StreamingScorer.from_pipeline factory."""

    def test_from_pipeline_creates_valid_scorer(self, fitted_pipeline):
        scorer = StreamingScorer.from_pipeline(fitted_pipeline)
        assert isinstance(scorer, StreamingScorer)
        window_size = fitted_pipeline.config.window.window_size
        assert scorer._window_size == window_size

    def test_from_pipeline_and_direct_init_equivalent(self, fitted_pipeline):
        scorer1 = StreamingScorer(fitted_pipeline)
        scorer2 = StreamingScorer.from_pipeline(fitted_pipeline)
        assert scorer1._window_size == scorer2._window_size


class TestScorePointShapeValidation:
    """Tests for H-3 fix: shape validation in score_point."""

    def test_scalar_input_accepted(self, fitted_pipeline):
        """Plain Python floats must be accepted without error."""
        scorer = StreamingScorer(fitted_pipeline)
        # Should not raise.
        result = scorer.score_point(0.5)
        assert result is None  # buffer not full yet on first call

    def test_zero_d_numpy_array_accepted(self, fitted_pipeline):
        """0-d numpy arrays are equivalent to scalars and must be accepted."""
        scorer = StreamingScorer(fitted_pipeline)
        result = scorer.score_point(np.float64(1.23))
        assert result is None

    def test_channel_mismatch_raises(self, fitted_pipeline):
        """Feeding a 2-channel value after a scalar must raise ValueError."""
        scorer = StreamingScorer(fitted_pipeline)
        scorer.score_point(0.1)  # establishes channel count = 1
        with pytest.raises(ValueError, match="Channel count mismatch"):
            scorer.score_point(np.array([0.1, 0.2]))  # 2 channels != 1

    def test_channel_mismatch_reversed_raises(self, fitted_pipeline):
        """Feeding a scalar after a 2-channel value must raise ValueError."""
        scorer = StreamingScorer(fitted_pipeline)
        scorer.score_point(np.array([0.1, 0.2]))  # establishes channel count = 2
        with pytest.raises(ValueError, match="Channel count mismatch"):
            scorer.score_point(0.3)  # 1 channel != 2

    def test_consistent_channels_do_not_raise(self, fitted_pipeline):
        """Consistent scalar inputs throughout must never raise."""
        scorer = StreamingScorer(fitted_pipeline)
        window_size = fitted_pipeline.config.window.window_size
        # Feed window_size + 1 consistent scalar values.
        result = None
        for i in range(window_size + 1):
            result = scorer.score_point(float(i) * 0.01)
        assert result is not None
        assert isinstance(result, float)

    def test_shape_error_message_contains_counts(self, fitted_pipeline):
        """Error message must report both expected and got channel counts."""
        scorer = StreamingScorer(fitted_pipeline)
        scorer.score_point(0.5)  # 1 channel
        with pytest.raises(ValueError, match=r"expected 1 channel"):
            scorer.score_point(np.array([0.1, 0.2, 0.3]))
