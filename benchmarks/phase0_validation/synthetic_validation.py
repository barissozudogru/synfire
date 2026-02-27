"""Phase 0 GO/NO-GO validation gate.

Trains a single FF layer on normal sine windows, then checks whether
goodness can separate normal vs anomalous windows. Uses Mann-Whitney U
to compute AUC-ROC without sklearn.

Pass criteria: AUC > 0.65
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

from synfire.core.config import FFLayerConfig, NormConfig, WindowConfig
from synfire.evaluation import mann_whitney_auc as _mw_auc
from synfire.layers.ff_layer import forward, goodness, init_layer, train_layer
from synfire.preprocessing.normalization import normalize_windows
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    sliding_windows,
)


def generate_sine_with_anomalies(
    length: int = 2000,
    period: int = 50,
    n_anomalies: int = 10,
    anomaly_magnitude: float = 5.0,
    seed: int = 42,
) -> tuple[NDArray, NDArray, list[int]]:
    """Generate a sine series with injected spike anomalies.

    Returns:
        (train_series, full_series, anomaly_positions)
        train_series is the first half (clean), full_series is all data.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    series = np.sin(2 * np.pi * t / period)

    split = length // 2
    candidates = range(split + 50, length - 10)
    anomaly_positions = sorted(rng.choice(candidates, n_anomalies, replace=False))

    for pos in anomaly_positions:
        series[pos : pos + 3] += anomaly_magnitude

    train_series = np.sin(2 * np.pi * np.arange(split, dtype=np.float64) / period)
    return train_series, series, anomaly_positions


def run_validation(verbose: bool = True) -> float:
    """Run the Phase 0 validation experiment.

    Returns:
        AUC-ROC score.
    """
    window_cfg = WindowConfig(window_size=20, stride=1)
    norm_cfg = NormConfig(method="zscore")

    # Generate data
    train_series, test_series, anomaly_positions = generate_sine_with_anomalies()

    # Prepare training windows (clean data only)
    train_windows = sliding_windows(train_series, window_cfg)
    train_windows = normalize_windows(train_windows, norm_cfg)

    rng = np.random.default_rng(42)
    pos_l, pos_r = make_consecutive_pairs(train_windows)
    neg_l, neg_r = make_random_pairs(train_windows, rng, min_gap=5)

    x_pos = np.concatenate([pos_l, pos_r], axis=1)
    x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

    # Train FF layer
    ff_cfg = FFLayerConfig(
        input_dim=40, hidden_dim=64, lr=0.01, threshold=2.0, epochs=200, seed=42
    )
    state = init_layer(ff_cfg)
    state, losses = train_layer(state, x_pos, x_neg)

    if verbose:
        print(f"Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Prepare test windows
    test_windows = sliding_windows(test_series, window_cfg)
    test_windows = normalize_windows(test_windows, norm_cfg)

    # Create test pairs (consecutive)
    test_pos_l, test_pos_r = make_consecutive_pairs(test_windows)
    test_pairs = np.concatenate([test_pos_l, test_pos_r], axis=1)

    # Get goodness for all test windows
    h = forward(state, test_pairs)
    g = goodness(h)

    # Label windows: anomaly if center of window is near an anomaly position
    anomaly_set = set()
    for pos in anomaly_positions:
        for offset in range(-window_cfg.window_size, window_cfg.window_size + 1):
            anomaly_set.add(pos + offset)

    labels = np.zeros(len(test_pos_l), dtype=bool)
    for i in range(len(test_pos_l)):
        window_center = i + window_cfg.window_size // 2
        if window_center in anomaly_set:
            labels[i] = True

    # Anomaly score = threshold - goodness (lower goodness = more anomalous)
    anomaly_scores = ff_cfg.threshold - g

    normal_scores = anomaly_scores[~labels]
    anomaly_scores_pos = anomaly_scores[labels]

    auc = _mw_auc(anomaly_scores, labels.astype(float))

    if verbose:
        print(f"Normal windows: {len(normal_scores)}, Anomaly windows: {len(anomaly_scores_pos)}")
        print(f"Mean goodness (normal): {np.mean(g[~labels]):.4f}")
        print(f"Mean goodness (anomaly): {np.mean(g[labels]):.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"{'GO' if auc > 0.65 else 'NO-GO'}: AUC {'>' if auc > 0.65 else '<='} 0.65")

    return auc


if __name__ == "__main__":
    auc = run_validation()
    sys.exit(0 if auc > 0.65 else 1)
