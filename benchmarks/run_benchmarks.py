"""Run synfire against baselines on all synthetic benchmark datasets.

Usage:
    poetry run python -m benchmarks.run_benchmarks
"""

from __future__ import annotations

import sys
import time

import numpy as np

from benchmarks.baselines import ZScoreBaseline
from benchmarks.metrics import auc_roc, best_f1, precision_at_k
from benchmarks.synthetic_datasets import ALL_DATASETS
from synfire import SynfireConfig, SynfirePipeline


def get_synfire_config() -> SynfireConfig:
    return SynfireConfig()  # uses tuned defaults from config.py


class SynfireAdapter:
    """Adapts SynfirePipeline to the baseline interface for benchmarking."""

    def __init__(self, config: SynfireConfig, seed: int = 42):
        from synfire.evaluation import _config_with_seed
        self._config = _config_with_seed(config, seed)
        self._pipeline = None

    def fit(self, series):
        self._pipeline = SynfirePipeline(self._config)
        self._pipeline.fit(series)

    def anomaly_scores(self, series):
        return self._pipeline.anomaly_scores(series)


def align_scores_labels(scores, labels):
    """Align score and label arrays to the shorter length."""
    n = min(len(scores), len(labels))
    return scores[:n], labels[:n]


def run_all(verbose: bool = True) -> dict:
    """Run all benchmarks and return results dict."""
    config = get_synfire_config()

    methods = {
        "synfire": SynfireAdapter(config, seed=42),
        "zscore_w20": ZScoreBaseline(window_size=20),
        "zscore_w50": ZScoreBaseline(window_size=50),
    }

    # Try to add sklearn baselines (requires optional benchmark deps)
    try:
        import sklearn  # noqa: F401

        from benchmarks.baselines import WindowedIsolationForest, WindowedLOF
        methods["iforest"] = WindowedIsolationForest(window_size=20)
        methods["lof"] = WindowedLOF(window_size=20, n_neighbors=15)
    except ImportError:
        pass

    results = {}
    all_aucs = {m: [] for m in methods}

    col_w = max(len(m) for m in methods)
    header = (
        f"{'Dataset':<20} | {'Method':<{col_w}} | "
        f"{'AUC':>6} | {'F1':>6} | {'P@k':>6} | {'Time':>7}"
    )
    sep = "-" * len(header)

    if verbose:
        print(header)
        print(sep)

    for ds_name, ds_fn in ALL_DATASETS.items():
        train, test, labels = ds_fn()

        for method_name, method in methods.items():
            t0 = time.perf_counter()
            method.fit(train)
            scores = method.anomaly_scores(test)
            elapsed = time.perf_counter() - t0

            s, lab = align_scores_labels(scores, labels)
            n_anom = max(int(lab.sum()), 10)

            auc = auc_roc(s, lab)
            f1 = best_f1(s, lab)
            pk = precision_at_k(s, lab, n_anom)

            all_aucs[method_name].append(auc)
            results[(ds_name, method_name)] = {
                "auc": auc, "f1": f1, "p@k": pk, "time": elapsed,
            }

            if verbose:
                print(
                    f"{ds_name:<20} | {method_name:<{col_w}} | "
                    f"{auc:>6.3f} | {f1:>6.3f} | {pk:>6.3f} | {elapsed:>6.2f}s"
                )

        if verbose:
            print(sep)

    if verbose:
        print(f"\n{'Method':<{col_w}} | {'Mean AUC':>8} | {'Std AUC':>7} | {'Wins':>4}")
        print("-" * (col_w + 30))

        # Count wins
        ds_names = list(ALL_DATASETS.keys())
        for method_name in methods:
            aucs = all_aucs[method_name]
            wins = 0
            for i, _ds in enumerate(ds_names):
                best_auc = max(all_aucs[m][i] for m in methods)
                if all_aucs[method_name][i] >= best_auc - 1e-10:
                    wins += 1
            print(
                f"{method_name:<{col_w}} | {np.mean(aucs):>8.4f} | "
                f"{np.std(aucs):>7.4f} | {wins:>4}"
            )

    return results


if __name__ == "__main__":
    results = run_all()
    # Exit 0 if synfire mean AUC > 0.65
    synfire_aucs = [v["auc"] for k, v in results.items() if k[1] == "synfire"]
    mean_auc = np.mean(synfire_aucs)
    print(f"\nSynfire mean AUC across all datasets: {mean_auc:.4f}")
    sys.exit(0 if mean_auc > 0.65 else 1)
