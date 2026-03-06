"""Evaluation utilities for multi-seed assessment and statistical testing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from synfire.api import SynfirePipeline
from synfire.core.config import SynfireConfig


def mann_whitney_auc(scores: NDArray, labels: NDArray) -> float:
    """Compute AUC-ROC via Mann-Whitney U statistic (O(N log N) sort-based).

    Uses rank-sum method to avoid the O(n_pos * n_neg) memory of the
    broadcasting approach, making it safe for large datasets.

    Args:
        scores: Anomaly scores of shape (N,). Higher = more anomalous.
        labels: Binary labels of shape (N,). 1 = anomaly, 0 = normal.

    Returns:
        AUC-ROC score in [0, 1].
    """
    labels_bool = labels.astype(bool)
    n_pos = int(labels_bool.sum())
    n_neg = int((~labels_bool).sum())

    if n_pos == 0 or n_neg == 0:
        return 0.5

    n = len(scores)
    # Stable sort: ties broken by original order (stable mergesort).
    order = np.argsort(scores, kind="mergesort")
    # Assign ranks 1..N; average tied ranks.
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Resolve ties by averaging ranks for equal score values.
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(sorted_scores[j] - sorted_scores[i]) < 1e-12:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j

    # U statistic = sum of positive ranks - n_pos*(n_pos+1)/2
    rank_sum_pos = float(ranks[labels_bool].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0

    return float(u / (n_pos * n_neg))


@dataclass
class EvaluationResult:
    """Results from multi-seed evaluation."""

    seeds: list[int]
    auc_scores: list[float]
    mean_auc: float = field(init=False)
    std_auc: float = field(init=False)

    def __post_init__(self):
        self.mean_auc = float(np.mean(self.auc_scores))
        self.std_auc = float(np.std(self.auc_scores))


def _config_with_seed(config: SynfireConfig, seed: int) -> SynfireConfig:
    """Create a copy of config with a different random seed, preserving all other settings."""
    return config.replace(ff_stack__seed=seed, hebbian__seed=seed)


def evaluate_multi_seed(
    train: NDArray,
    test: NDArray,
    labels: NDArray,
    config: SynfireConfig | None = None,
    n_seeds: int = 5,
    base_seed: int = 0,
) -> EvaluationResult:
    """Run evaluation across multiple random seeds and report statistics.

    Args:
        train: Training time series (1D or 2D).
        test: Test time series (1D or 2D).
        labels: Binary anomaly labels for the test windows.
            Shape (N_windows - 1,) matching anomaly_scores output.
        config: Base config. Defaults will be used if None.
        n_seeds: Number of random seeds to evaluate.
        base_seed: Starting seed value.

    Returns:
        EvaluationResult with per-seed AUCs and summary statistics.
    """
    if config is None:
        config = SynfireConfig()

    seeds = list(range(base_seed, base_seed + n_seeds))
    auc_scores = []

    for seed in seeds:
        seeded_config = _config_with_seed(config, seed)
        pipeline = SynfirePipeline(seeded_config)
        pipeline.fit(train)
        scores = pipeline.anomaly_scores(test)

        # Align labels and scores to the shorter length
        n = min(len(scores), len(labels))
        auc = mann_whitney_auc(scores[:n], labels[:n])
        auc_scores.append(auc)

    return EvaluationResult(seeds=seeds, auc_scores=auc_scores)
