"""Benchmark runner: evaluate multiple methods on multiple datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from benchmarks.metrics import auc_roc, best_f1, precision_at_k


@dataclass
class BenchmarkResult:
    """Results for one method on one dataset."""

    method_name: str
    dataset_name: str
    auc: float
    f1: float
    precision_at_10: float
    n_seeds: int = 1


def run_benchmark(
    datasets: dict[str, tuple[NDArray, NDArray, NDArray]],
    methods: dict[str, object],
    n_seeds: int = 1,
) -> list[BenchmarkResult]:
    """Run all methods on all datasets and collect results.

    Args:
        datasets: {name: (train_series, test_series, test_labels)} dict.
        methods: {name: detector_instance} dict. Each detector must have
            fit(series) and anomaly_scores(series) methods.
        n_seeds: Number of seeds (only relevant for stochastic methods).

    Returns:
        List of BenchmarkResult, one per (method, dataset) pair.
    """
    results = []

    for ds_name, (train, test, labels) in datasets.items():
        for method_name, detector in methods.items():
            aucs = []
            f1s = []
            p10s = []

            for seed in range(n_seeds):
                if hasattr(detector, "seed"):
                    detector.seed = seed

                detector.fit(train)
                scores = detector.anomaly_scores(test)

                # Align labels with scores (methods may produce different lengths)
                n = min(len(scores), len(labels))
                s = scores[:n]
                lab = labels[:n]

                n_anomalies = int(lab.sum())
                k = max(n_anomalies, 10)

                aucs.append(auc_roc(s, lab))
                f1s.append(best_f1(s, lab))
                p10s.append(precision_at_k(s, lab, k))

            results.append(BenchmarkResult(
                method_name=method_name,
                dataset_name=ds_name,
                auc=float(np.mean(aucs)),
                f1=float(np.mean(f1s)),
                precision_at_10=float(np.mean(p10s)),
                n_seeds=n_seeds,
            ))

    return results


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as an ASCII comparison table."""
    if not results:
        print("No results to display.")
        return

    # Group by dataset
    datasets = sorted(set(r.dataset_name for r in results))
    methods = sorted(set(r.method_name for r in results))

    # Header
    method_width = max(len(m) for m in methods)
    col_width = max(method_width, 8)
    header = f"{'Dataset':<25} | {'Method':<{col_width}} | {'AUC':>6} | {'F1':>6} | {'P@k':>6}"
    print(header)
    print("-" * len(header))

    for ds in datasets:
        ds_results = [r for r in results if r.dataset_name == ds]
        for r in sorted(ds_results, key=lambda x: x.method_name):
            print(
                f"{r.dataset_name:<25} | {r.method_name:<{col_width}} | "
                f"{r.auc:>6.3f} | {r.f1:>6.3f} | {r.precision_at_10:>6.3f}"
            )
        print("-" * len(header))
