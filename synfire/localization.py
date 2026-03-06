"""Anomaly localization: segment detection and merging from raw anomaly scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AnomalySegment:
    """A contiguous anomalous region in the score sequence.

    Attributes:
        start_idx: Inclusive start index in the score array.
        end_idx: Exclusive end index in the score array.
        peak_score: Maximum score within the segment.
        mean_score: Mean score within the segment.
    """

    start_idx: int
    end_idx: int
    peak_score: float
    mean_score: float

    def __len__(self) -> int:
        return self.end_idx - self.start_idx


def localize_anomalies(
    scores: NDArray,
    threshold: float | None = None,
    percentile: float | None = None,
    min_duration: int = 1,
    merge_gap: int = 0,
) -> list[AnomalySegment]:
    """Detect contiguous anomalous runs in a score array.

    Exactly one of ``threshold`` or ``percentile`` must be provided.

    Args:
        scores: 1D array of anomaly scores (higher = more anomalous).
        threshold: Fixed score threshold; samples with score >= threshold are
            considered anomalous.
        percentile: Compute the threshold as ``np.percentile(scores, percentile)``.
            For example, ``percentile=95`` flags the top 5% of scores.
        min_duration: Minimum number of consecutive flagged samples required to
            keep a segment. Shorter segments are discarded.
        merge_gap: Maximum gap (number of below-threshold samples) between two
            adjacent segments to merge them into one.

    Returns:
        Sorted list of ``AnomalySegment`` objects.

    Raises:
        ValueError: If neither or both of ``threshold`` / ``percentile`` are given,
            or if inputs are invalid.
    """
    if threshold is None and percentile is None:
        raise ValueError("Provide exactly one of 'threshold' or 'percentile'.")
    if threshold is not None and percentile is not None:
        raise ValueError("Provide exactly one of 'threshold' or 'percentile', not both.")

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got ndim={scores.ndim}")

    if percentile is not None:
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        threshold = float(np.percentile(scores, percentile))
    else:
        threshold = float(threshold)  # type: ignore[arg-type]

    if min_duration < 1:
        raise ValueError(f"min_duration must be >= 1, got {min_duration}")
    if merge_gap < 0:
        raise ValueError(f"merge_gap must be >= 0, got {merge_gap}")

    # Find contiguous runs where score >= threshold
    flagged = scores >= threshold
    segments = _find_runs(flagged)

    # Merge close segments
    if merge_gap > 0 and len(segments) > 1:
        segments = _merge_segments(segments, merge_gap)

    # Filter by minimum duration
    segments = [s for s in segments if len(s) >= min_duration]

    # Attach score statistics
    result = []
    for seg in segments:
        chunk = scores[seg.start_idx : seg.end_idx]
        result.append(
            AnomalySegment(
                start_idx=seg.start_idx,
                end_idx=seg.end_idx,
                peak_score=float(chunk.max()),
                mean_score=float(chunk.mean()),
            )
        )

    return result


def _find_runs(flagged: NDArray) -> list[AnomalySegment]:
    """Find contiguous True-runs in a boolean array."""
    segments: list[AnomalySegment] = []
    n = len(flagged)
    i = 0
    while i < n:
        if flagged[i]:
            j = i + 1
            while j < n and flagged[j]:
                j += 1
            segments.append(AnomalySegment(start_idx=i, end_idx=j, peak_score=0.0, mean_score=0.0))
            i = j
        else:
            i += 1
    return segments


def _merge_segments(
    segments: list[AnomalySegment], merge_gap: int
) -> list[AnomalySegment]:
    """Merge segments whose gap is <= merge_gap."""
    merged = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        gap = current.start_idx - prev.end_idx
        if gap <= merge_gap:
            merged[-1] = AnomalySegment(
                start_idx=prev.start_idx,
                end_idx=current.end_idx,
                peak_score=0.0,
                mean_score=0.0,
            )
        else:
            merged.append(current)
    return merged
