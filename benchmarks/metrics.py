"""Benchmark metrics: AUC-ROC, precision@k, best F1 -- all pure NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synfire.evaluation import mann_whitney_auc


def auc_roc(scores: NDArray, labels: NDArray) -> float:
    """Compute AUC-ROC via Mann-Whitney U statistic.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Binary labels (1 = anomaly).

    Returns:
        AUC-ROC in [0, 1].
    """
    return mann_whitney_auc(scores, labels)


def precision_at_k(scores: NDArray, labels: NDArray, k: int) -> float:
    """Precision among the top-k scored samples.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Binary labels (1 = anomaly).
        k: Number of top scores to consider.

    Returns:
        Fraction of true anomalies in the top-k.
    """
    if k <= 0:
        return 0.0
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[-k:]
    return float(labels[top_k_idx].sum() / k)


def best_f1(scores: NDArray, labels: NDArray, n_thresholds: int = 200) -> float:
    """Find the best F1 score over a range of thresholds.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Binary labels (1 = anomaly).
        n_thresholds: Number of threshold values to try.

    Returns:
        Best F1 score in [0, 1].
    """
    labels_bool = labels.astype(bool)
    n_pos = labels_bool.sum()
    if n_pos == 0:
        return 0.0

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best = 0.0

    for t in thresholds:
        predicted = scores >= t
        tp = (predicted & labels_bool).sum()
        fp = (predicted & ~labels_bool).sum()
        fn = (~predicted & labels_bool).sum()

        if tp == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1

    return float(best)
