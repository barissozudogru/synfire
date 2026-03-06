"""Phase 0 pipeline validation: full SynfirePipeline with multi-seed evaluation.

Tests the complete pipeline (not just a single FF layer) against synthetic
data with known anomalies. Reports mean AUC +/- std across seeds, runs
ablation per scoring component, and compares against z-score baseline.

Pass criteria: mean AUC > 0.70
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

from benchmarks.baselines import ZScoreBaseline
from benchmarks.metrics import auc_roc
from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import (
    AnomalyConfig,
    FFStackConfig,
    HebbianConfig,
    NormConfig,
    WindowConfig,
)
from synfire.evaluation import evaluate_multi_seed


def generate_validation_data(
    length: int = 2000,
    period: int = 50,
    n_anomalies: int = 10,
    anomaly_magnitude: float = 5.0,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray, list[int]]:
    """Generate sine series with spike anomalies and window-level labels.

    Returns:
        (train_series, test_series, window_labels, anomaly_positions)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    series = np.sin(2 * np.pi * t / period)

    split = length // 2
    candidates = list(range(split + 50, length - 10))
    anomaly_positions = sorted(rng.choice(candidates, n_anomalies, replace=False))

    for pos in anomaly_positions:
        series[pos : pos + 3] += anomaly_magnitude

    train_series = np.sin(2 * np.pi * np.arange(split, dtype=np.float64) / period)
    test_series = series

    # Build window-level labels
    window_size = 20
    n_windows = len(test_series) - window_size
    labels = np.zeros(n_windows)

    anomaly_set = set()
    for pos in anomaly_positions:
        for offset in range(-window_size, window_size + 1):
            anomaly_set.add(pos + offset)

    for i in range(n_windows):
        center = i + window_size // 2
        if center in anomaly_set:
            labels[i] = 1

    return train_series, test_series, labels, anomaly_positions


def get_base_config() -> SynfireConfig:
    """Standard config for Phase 0 validation."""
    return SynfireConfig(
        window=WindowConfig(window_size=20, stride=1),
        norm=NormConfig(method="zscore"),
        ff_stack=FFStackConfig(
            layer_dims=(64, 32), lr=0.01, threshold=2.0, epochs_per_layer=100, seed=42,
        ),
        hebbian=HebbianConfig(
            n_prototypes=8, lr=0.01, inhibition_strength=0.1, epochs=50, seed=42,
        ),
        anomaly=AnomalyConfig(
            weight_goodness=0.5, weight_distance=0.3, weight_transition=0.2,
        ),
    )


def run_multi_seed_validation(verbose: bool = True) -> float:
    """Run full pipeline validation with 5 seeds."""
    train, test, labels, _ = generate_validation_data()
    config = get_base_config()

    result = evaluate_multi_seed(
        train, test, labels, config=config, n_seeds=5, base_seed=0,
    )

    if verbose:
        print("=== Multi-Seed Pipeline Validation ===")
        for seed, auc in zip(result.seeds, result.auc_scores, strict=True):
            print(f"  Seed {seed}: AUC = {auc:.4f}")
        print(f"  Mean AUC: {result.mean_auc:.4f} +/- {result.std_auc:.4f}")
        verdict = "PASS" if result.mean_auc > 0.70 else "FAIL"
        cmp = ">" if result.mean_auc > 0.70 else "<="
        print(f"  {verdict}: mean AUC {cmp} 0.70")

    return result.mean_auc


def run_zscore_comparison(verbose: bool = True) -> float:
    """Run z-score baseline for comparison."""
    train, test, labels, _ = generate_validation_data()

    baseline = ZScoreBaseline(window_size=50)
    baseline.fit(train)
    scores = baseline.anomaly_scores(test)

    # Align with window labels
    n = min(len(scores), len(labels))
    auc = auc_roc(scores[:n], labels[:n])

    if verbose:
        print("\n=== Z-Score Baseline ===")
        print(f"  AUC: {auc:.4f}")

    return auc


def run_ablation(verbose: bool = True) -> dict[str, float]:
    """Run ablation: each scoring component solo."""
    train, test, labels, _ = generate_validation_data()
    base = get_base_config()

    components = {
        "goodness_only": AnomalyConfig(
            weight_goodness=1.0, weight_distance=0.0, weight_transition=0.0,
            use_goodness=True, use_distance=False, use_transition=False,
        ),
        "distance_only": AnomalyConfig(
            weight_goodness=0.0, weight_distance=1.0, weight_transition=0.0,
            use_goodness=False, use_distance=True, use_transition=False,
        ),
        "transition_only": AnomalyConfig(
            weight_goodness=0.0, weight_distance=0.0, weight_transition=1.0,
            use_goodness=False, use_distance=False, use_transition=True,
        ),
    }

    results = {}

    if verbose:
        print("\n=== Ablation Study ===")

    for name, anomaly_config in components.items():
        config = SynfireConfig(
            window=base.window,
            norm=base.norm,
            ff_stack=base.ff_stack,
            hebbian=base.hebbian,
            anomaly=anomaly_config,
        )

        pipeline = SynfirePipeline(config)
        pipeline.fit(train)
        scores = pipeline.anomaly_scores(test)

        n = min(len(scores), len(labels))
        auc = auc_roc(scores[:n], labels[:n])
        results[name] = auc

        if verbose:
            print(f"  {name}: AUC = {auc:.4f}")

    return results


def run_validation(verbose: bool = True) -> bool:
    """Run complete Phase 0 validation suite.

    Returns:
        True if validation passes (mean AUC > 0.70).
    """
    mean_auc = run_multi_seed_validation(verbose)
    zscore_auc = run_zscore_comparison(verbose)
    ablation = run_ablation(verbose)

    passed = mean_auc > 0.70

    if verbose:
        print("\n=== Summary ===")
        print(f"  Pipeline mean AUC: {mean_auc:.4f}")
        print(f"  Z-Score baseline AUC: {zscore_auc:.4f}")
        print(f"  Ablation: {', '.join(f'{k}={v:.3f}' for k, v in ablation.items())}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


if __name__ == "__main__":
    passed = run_validation()
    sys.exit(0 if passed else 1)
