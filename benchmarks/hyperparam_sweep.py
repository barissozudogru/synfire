"""Hyperparameter sensitivity analysis for synfire.

Sweeps key parameters one at a time while holding others at defaults,
reporting AUC impact on the sine_spike dataset.

Usage:
    poetry run python -m benchmarks.hyperparam_sweep
"""

from __future__ import annotations

import numpy as np

from benchmarks.metrics import auc_roc
from benchmarks.synthetic_datasets import sine_spike
from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import (
    AnomalyConfig,
    FFStackConfig,
    HebbianConfig,
    WindowConfig,
)
from synfire.evaluation import _config_with_seed


def _evaluate(config: SynfireConfig, train, test, labels, n_seeds=3):
    """Evaluate config across multiple seeds, return mean AUC."""
    aucs = []
    for seed in range(n_seeds):
        c = _config_with_seed(config, seed)
        p = SynfirePipeline(c)
        p.fit(train)
        scores = p.anomaly_scores(test)
        n = min(len(scores), len(labels))
        aucs.append(auc_roc(scores[:n], labels[:n]))
    return np.mean(aucs), np.std(aucs)


def sweep_window_size(train, test, labels):
    """Sweep window_size: 10, 15, 20, 25, 30, 40, 50."""
    values = [10, 15, 20, 25, 30, 40, 50]
    results = []
    for ws in values:
        config = SynfireConfig(
            window=WindowConfig(window_size=ws, stride=1),
            ff_stack=FFStackConfig(layer_dims=(64, 32), lr=0.01, epochs_per_layer=100),
            hebbian=HebbianConfig(n_prototypes=8, lr=0.01, epochs=50),
        )
        mean, std = _evaluate(config, train, test, labels)
        results.append((ws, mean, std))
    return "window_size", results


def sweep_n_prototypes(train, test, labels):
    """Sweep n_prototypes: 2, 4, 8, 12, 16."""
    values = [2, 4, 8, 12, 16]
    results = []
    for np_ in values:
        config = SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(64, 32), lr=0.01, epochs_per_layer=100),
            hebbian=HebbianConfig(n_prototypes=np_, lr=0.01, epochs=50),
        )
        mean, std = _evaluate(config, train, test, labels)
        results.append((np_, mean, std))
    return "n_prototypes", results


def sweep_layer_dims(train, test, labels):
    """Sweep layer architectures."""
    architectures = [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ]
    results = []
    for dims in architectures:
        config = SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=dims, lr=0.01, epochs_per_layer=100),
            hebbian=HebbianConfig(n_prototypes=8, lr=0.01, epochs=50),
        )
        mean, std = _evaluate(config, train, test, labels)
        results.append((str(dims), mean, std))
    return "layer_dims", results


def sweep_ff_lr(train, test, labels):
    """Sweep FF learning rate."""
    values = [0.001, 0.005, 0.01, 0.02, 0.05]
    results = []
    for lr in values:
        config = SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(64, 32), lr=lr, epochs_per_layer=100),
            hebbian=HebbianConfig(n_prototypes=8, lr=0.01, epochs=50),
        )
        mean, std = _evaluate(config, train, test, labels)
        results.append((lr, mean, std))
    return "ff_lr", results


def sweep_ff_epochs(train, test, labels):
    """Sweep FF epochs per layer."""
    values = [30, 50, 100, 150, 200]
    results = []
    for ep in values:
        config = SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(64, 32), lr=0.01, epochs_per_layer=ep),
            hebbian=HebbianConfig(n_prototypes=8, lr=0.01, epochs=50),
        )
        mean, std = _evaluate(config, train, test, labels)
        results.append((ep, mean, std))
    return "ff_epochs_per_layer", results


def sweep_weights(train, test, labels):
    """Sweep anomaly scoring weight combinations."""
    combos = [
        (1.0, 0.0, 0.0, "goodness_only"),
        (0.0, 1.0, 0.0, "distance_only"),
        (0.0, 0.0, 1.0, "transition_only"),
        (0.5, 0.5, 0.0, "g+d"),
        (0.5, 0.0, 0.5, "g+t"),
        (0.0, 0.5, 0.5, "d+t"),
        (0.5, 0.3, 0.2, "default"),
        (0.4, 0.4, 0.2, "equal_gd"),
        (0.3, 0.5, 0.2, "distance_heavy"),
        (0.2, 0.6, 0.2, "distance_dominant"),
    ]
    results = []
    for wg, wd, wt, label in combos:
        config = SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(64, 32), lr=0.01, epochs_per_layer=100),
            hebbian=HebbianConfig(n_prototypes=8, lr=0.01, epochs=50),
            anomaly=AnomalyConfig(
                weight_goodness=wg, weight_distance=wd, weight_transition=wt,
                use_goodness=wg > 0, use_distance=wd > 0, use_transition=wt > 0,
            ),
        )
        mean, std = _evaluate(config, train, test, labels)
        results.append((label, mean, std))
    return "score_weights", results


def run_sweep():
    """Run full hyperparameter sweep."""
    train, test, labels = sine_spike(seed=42)

    print("=" * 60)
    print("Hyperparameter Sensitivity Analysis (sine_spike dataset)")
    print("=" * 60)

    sweeps = [
        sweep_window_size,
        sweep_n_prototypes,
        sweep_layer_dims,
        sweep_ff_lr,
        sweep_ff_epochs,
        sweep_weights,
    ]

    best_overall = {}

    for sweep_fn in sweeps:
        name, results = sweep_fn(train, test, labels)
        print(f"\n--- {name} ---")
        print(f"  {'Value':<20} | {'Mean AUC':>8} | {'Std':>6}")
        print(f"  {'-' * 40}")

        best_val, best_auc = None, 0
        for val, mean, std in results:
            marker = ""
            if mean > best_auc:
                best_auc = mean
                best_val = val
                marker = " <-- best"
            print(f"  {str(val):<20} | {mean:>8.4f} | {std:>6.4f}{marker}")

        # Reprint best
        print(f"  Best: {best_val} (AUC={best_auc:.4f})")
        best_overall[name] = (best_val, best_auc)

    print("\n" + "=" * 60)
    print("Summary of best values:")
    for name, (val, auc) in best_overall.items():
        print(f"  {name}: {val} (AUC={auc:.4f})")


if __name__ == "__main__":
    run_sweep()
