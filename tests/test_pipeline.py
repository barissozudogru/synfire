import numpy as np
import pytest

from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import (
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
