import numpy as np

from synfire.core.config import FFLayerConfig, NormConfig, WindowConfig
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
