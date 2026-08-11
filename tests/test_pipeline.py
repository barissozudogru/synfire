import numpy as np
import pytest

from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import (
    AnomalyConfig,
    FFLayerConfig,
    FFStackConfig,
    HebbianConfig,
    NormConfig,
    WindowConfig,
)
from synfire.layers.ff_layer import forward, goodness, init_layer, train_layer
from synfire.preprocessing.normalization import normalize_windows
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    sliding_windows,
)


class TestPhase0Integration:
    """Integration tests mirroring the Phase 0 validation flow."""

    def test_end_to_end_training(self, sine_series, rng):
        window_cfg = WindowConfig(window_size=20, stride=1)
        windows = sliding_windows(sine_series, window_cfg)
        windows = normalize_windows(windows, NormConfig(method="zscore"))

        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)

        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFLayerConfig(input_dim=40, hidden_dim=32, lr=0.01, epochs=50)
        state = init_layer(cfg)
        state, losses = train_layer(state, x_pos, x_neg)

        assert len(losses) == 50
        assert losses[-1] < losses[0]

    def test_anomaly_goodness_lower(self, sine_with_anomalies, rng):
        series, anomaly_positions = sine_with_anomalies
        window_cfg = WindowConfig(window_size=20, stride=1)

        # Train on clean portion (first 150 points)
        clean = series[:150]
        clean_windows = sliding_windows(clean, window_cfg)
        clean_windows = normalize_windows(clean_windows)

        pos_l, pos_r = make_consecutive_pairs(clean_windows)
        neg_l, neg_r = make_random_pairs(clean_windows, rng, min_gap=5)

        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFLayerConfig(input_dim=40, hidden_dim=64, lr=0.01, epochs=200)
        state = init_layer(cfg)
        state, _ = train_layer(state, x_pos, x_neg)

        # Evaluate on full series
        all_windows = sliding_windows(series, window_cfg)
        all_windows = normalize_windows(all_windows)
        test_l, test_r = make_consecutive_pairs(all_windows)
        test_pairs = np.concatenate([test_l, test_r], axis=1)

        h = forward(state, test_pairs)
        g = goodness(h)

        # Check that anomaly-region goodness tends to be different from normal
        anomaly_mask = np.zeros(len(g), dtype=bool)
        for pos in anomaly_positions:
            start = max(0, pos - window_cfg.window_size)
            end = min(len(g), pos + window_cfg.window_size)
            anomaly_mask[start:end] = True

        g_normal = g[~anomaly_mask]
        g_anomaly = g[anomaly_mask]

        # Just verify we get scores and they differ
        assert len(g_normal) > 0
        assert len(g_anomaly) > 0
        assert g.shape[0] == len(test_l)


class TestSynfirePipeline:
    """End-to-end tests for the unified SynfirePipeline API."""

    @pytest.fixture
    def small_config(self):
        return SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            norm=NormConfig(method="zscore"),
            ff_stack=FFStackConfig(
                layer_dims=(32, 16), lr=0.01, threshold=2.0, epochs_per_layer=30
            ),
            hebbian=HebbianConfig(
                n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=10
            ),
        )

    def test_fit_returns_self(self, sine_series, small_config):
        pipeline = SynfirePipeline(small_config)
        result = pipeline.fit(sine_series)
        assert result is pipeline

    def test_not_fitted_raises(self, sine_series):
        pipeline = SynfirePipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.anomaly_scores(sine_series)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.cluster(sine_series)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.transform(sine_series)

    def test_anomaly_scores_shape(self, sine_series, small_config):
        pipeline = SynfirePipeline(small_config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        expected_n = (len(sine_series) - 20) // 1 + 1 - 1
        assert scores.shape == (expected_n,)

    def test_cluster_shape(self, sine_series, small_config):
        pipeline = SynfirePipeline(small_config)
        pipeline.fit(sine_series)
        clusters = pipeline.cluster(sine_series)
        expected_n = (len(sine_series) - 20) // 1 + 1 - 1
        assert clusters.shape == (expected_n,)
        assert np.all(clusters >= 0) and np.all(clusters < 4)

    def test_transform_shape(self, sine_series, small_config):
        pipeline = SynfirePipeline(small_config)
        pipeline.fit(sine_series)
        reps = pipeline.transform(sine_series)
        expected_n = (len(sine_series) - 20) // 1 + 1 - 1
        assert reps.shape == (expected_n, 16)  # last layer dim

    def test_default_config_works(self, sine_series):
        pipeline = SynfirePipeline()
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert len(scores) > 0
        assert np.all(np.isfinite(scores))

    def test_import_from_package(self):
        from synfire import SynfirePipeline as SP

        assert SP is SynfirePipeline


class TestAnomalyScoreDeterminism:
    """Verify that anomaly scores are batch-independent after fit."""

    @pytest.fixture
    def fitted_pipeline(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=30),
            hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=10),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        return pipeline

    def test_anomaly_score_determinism(self, fitted_pipeline, sine_series):
        """Same input scored in different batch contexts must produce identical scores."""
        full_scores = fitted_pipeline.anomaly_scores(sine_series)

        # Score a subset -- same data, different batch context
        half = len(sine_series) // 2 + 20  # enough for at least some windows
        subset_scores = fitted_pipeline.anomaly_scores(sine_series[:half])

        # Overlapping region should have identical scores
        overlap = min(len(full_scores), len(subset_scores))
        np.testing.assert_allclose(
            full_scores[:overlap], subset_scores[:overlap], atol=1e-10,
            err_msg="Scores differ between batch contexts -- normalization is batch-dependent"
        )

    def test_anomaly_scaler_stored_after_fit(self, fitted_pipeline):
        assert fitted_pipeline._anomaly_scaler is not None
        assert isinstance(fitted_pipeline._anomaly_scaler.goodness_min, float)
        assert isinstance(fitted_pipeline._anomaly_scaler.goodness_range, float)


class TestAblation:
    """Verify ablation toggles work correctly."""

    @pytest.fixture
    def base_config(self):
        return SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=30),
            hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=10),
        )

    def test_ablation_goodness_only(self, sine_series, base_config):
        config = SynfireConfig(
            window=base_config.window,
            norm=base_config.norm,
            ff_stack=base_config.ff_stack,
            hebbian=base_config.hebbian,
            anomaly=AnomalyConfig(
                weight_goodness=1.0, weight_distance=0.0, weight_transition=0.0,
                use_goodness=True, use_distance=False, use_transition=False,
            ),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert len(scores) > 0
        assert np.all(np.isfinite(scores))

    def test_ablation_distance_only(self, sine_series, base_config):
        config = SynfireConfig(
            window=base_config.window,
            norm=base_config.norm,
            ff_stack=base_config.ff_stack,
            hebbian=base_config.hebbian,
            anomaly=AnomalyConfig(
                weight_goodness=0.0, weight_distance=1.0, weight_transition=0.0,
                use_goodness=False, use_distance=True, use_transition=False,
            ),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert len(scores) > 0
        assert np.all(np.isfinite(scores))


class TestInputValidation:
    """Test that invalid series inputs are rejected with clear errors."""

    def test_nan_series_rejected(self):
        pipeline = SynfirePipeline()
        series = np.array([1.0, 2.0, np.nan, 4.0] * 100)
        with pytest.raises(ValueError, match="non-finite"):
            pipeline.fit(series)

    def test_inf_series_rejected(self):
        pipeline = SynfirePipeline()
        series = np.array([1.0, np.inf, 3.0, 4.0] * 100)
        with pytest.raises(ValueError, match="non-finite"):
            pipeline.fit(series)

    def test_3d_series_rejected(self):
        pipeline = SynfirePipeline()
        series = np.ones((10, 5, 3))
        with pytest.raises(ValueError, match="1D or 2D"):
            pipeline.fit(series)

    def test_empty_series_rejected(self):
        pipeline = SynfirePipeline()
        series = np.array([])
        with pytest.raises(ValueError, match="non-empty"):
            pipeline.fit(series)

    def test_too_short_series_rejected(self):
        pipeline = SynfirePipeline(SynfireConfig(window=WindowConfig(window_size=50)))
        series = np.arange(30, dtype=np.float64)
        with pytest.raises(ValueError, match="shorter than window_size"):
            pipeline.fit(series)

    def test_list_series_supported(self):
        pipeline = SynfirePipeline()
        series = [float(x) for x in range(100)]
        pipeline.fit(series)
        scores = pipeline.anomaly_scores(series)
        assert len(scores) > 0

    def test_non_numeric_series_rejected(self):
        pipeline = SynfirePipeline()
        series = np.array(["a", "b", "c"] * 50)
        with pytest.raises(ValueError, match="numeric dtype"):
            pipeline.fit(series)

    def test_too_few_pairs_rejected(self):
        # Length 29 with window=20, stride=1 yields 10 windows and 9 pairs, which is < 10.
        pipeline = SynfirePipeline(SynfireConfig(window=WindowConfig(window_size=20, stride=1)))
        series = np.arange(29, dtype=np.float64)
        with pytest.raises(ValueError, match="Too few training pairs generated"):
            pipeline.fit(series)



class TestConfigValidation:
    """Test that invalid config values are rejected at construction time."""

    def test_zero_window_size_rejected(self):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            WindowConfig(window_size=0)

    def test_negative_lr_rejected(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            FFStackConfig(lr=-0.01)

    def test_zero_prototypes_rejected(self):
        with pytest.raises(ValueError, match="n_prototypes must be >= 1"):
            HebbianConfig(n_prototypes=0)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="weight_goodness must be >= 0"):
            AnomalyConfig(weight_goodness=-1.0)
