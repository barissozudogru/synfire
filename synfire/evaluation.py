"""Evaluation utilities for multi-seed assessment and statistical testing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from synfire.api import SynfirePipeline
from synfire.core.config import FFStackConfig, HebbianConfig, SynfireConfig


def mann_whitney_auc(scores: NDArray, labels: NDArray) -> float:
    """Compute AUC-ROC via Mann-Whitney U statistic.

    Args:
        scores: Anomaly scores of shape (N,). Higher = more anomalous.
        labels: Binary labels of shape (N,). 1 = anomaly, 0 = normal.

    Returns:
        AUC-ROC score in [0, 1].
    """
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Vectorized: count how often pos > neg
    # Using broadcasting: (n_pos, 1) vs (1, n_neg)
    comparisons = pos[:, np.newaxis] > neg[np.newaxis, :]
    ties = pos[:, np.newaxis] == neg[np.newaxis, :]
    u = comparisons.sum() + 0.5 * ties.sum()

    return float(u / (len(pos) * len(neg)))


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
    """Create a copy of config with a different random seed."""
    return SynfireConfig(
        window=config.window,
        norm=config.norm,
        ff_stack=FFStackConfig(
            layer_dims=config.ff_stack.layer_dims,
            lr=config.ff_stack.lr,
            threshold=config.ff_stack.threshold,
            epochs_per_layer=config.ff_stack.epochs_per_layer,
            seed=seed,
        ),
        hebbian=HebbianConfig(
            n_prototypes=config.hebbian.n_prototypes,
            lr=config.hebbian.lr,
            inhibition_strength=config.hebbian.inhibition_strength,
            epochs=config.hebbian.epochs,
            seed=seed,
        ),
        anomaly=config.anomaly,
    )


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
