"""Tests for synfire.localization: anomaly segment detection and merging."""

from __future__ import annotations

import numpy as np
import pytest

from synfire.localization import localize_anomalies


class TestLocalizeAnomaliesFixedThreshold:
    """Tests using a fixed numeric threshold."""

    def test_single_segment_detected(self):
        scores = np.array([0.1, 0.2, 0.9, 0.95, 0.8, 0.1, 0.2], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.7)
        assert len(segments) == 1
        seg = segments[0]
        assert seg.start_idx == 2
        assert seg.end_idx == 5

    def test_segment_peak_and_mean_scores(self):
        scores = np.array([0.0, 1.0, 2.0, 1.5, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        assert len(segments) == 1
        seg = segments[0]
        assert seg.peak_score == pytest.approx(2.0)
        assert seg.mean_score == pytest.approx(np.mean([1.0, 2.0, 1.5]))

    def test_two_separate_segments(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        assert len(segments) == 2
        assert segments[0].start_idx == 1
        assert segments[0].end_idx == 3
        assert segments[1].start_idx == 5
        assert segments[1].end_idx == 7

    def test_all_above_threshold(self):
        scores = np.ones(10, dtype=np.float64) * 2.0
        segments = localize_anomalies(scores, threshold=1.0)
        assert len(segments) == 1
        assert segments[0].start_idx == 0
        assert segments[0].end_idx == 10

    def test_all_below_threshold(self):
        scores = np.zeros(10, dtype=np.float64)
        segments = localize_anomalies(scores, threshold=1.0)
        assert len(segments) == 0

    def test_exact_threshold_boundary_inclusive(self):
        scores = np.array([0.5, 1.0, 0.5], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=1.0)
        assert len(segments) == 1
        assert segments[0].start_idx == 1
        assert segments[0].end_idx == 2

    def test_segment_length_property(self):
        scores = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        assert len(segments) == 1
        assert len(segments[0]) == 3


class TestLocalizeAnomaliesPercentileThreshold:
    """Tests using a percentile-derived threshold."""

    def test_percentile_95_top_five_percent(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 200).astype(np.float64)
        segments = localize_anomalies(scores, percentile=95.0)
        # At least some high-scoring points should be flagged
        total_flagged = sum(len(s) for s in segments)
        assert total_flagged > 0

    def test_percentile_0_all_above(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        segments = localize_anomalies(scores, percentile=0.0)
        total_flagged = sum(len(s) for s in segments)
        assert total_flagged == len(scores)

    def test_percentile_100_single_or_none(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        segments = localize_anomalies(scores, percentile=100.0)
        total_flagged = sum(len(s) for s in segments)
        # Only values equal to the maximum qualify
        assert total_flagged >= 1

    def test_percentile_and_threshold_both_raises(self):
        scores = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="exactly one"):
            localize_anomalies(scores, threshold=0.5, percentile=90.0)

    def test_neither_raises(self):
        scores = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="exactly one"):
            localize_anomalies(scores)

    def test_invalid_percentile_raises(self):
        scores = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="percentile must be in"):
            localize_anomalies(scores, percentile=101.0)


class TestMinDurationFiltering:
    """Tests for min_duration parameter."""

    def test_min_duration_filters_short_segments(self):
        # Two segments: one of length 1, one of length 3
        scores = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, min_duration=2)
        assert len(segments) == 1
        assert segments[0].start_idx == 4
        assert segments[0].end_idx == 7

    def test_min_duration_1_keeps_all(self):
        scores = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, min_duration=1)
        assert len(segments) == 2

    def test_min_duration_exceeds_all_segments(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, min_duration=5)
        assert len(segments) == 0

    def test_invalid_min_duration_raises(self):
        scores = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="min_duration must be >= 1"):
            localize_anomalies(scores, threshold=0.5, min_duration=0)


class TestMergeGap:
    """Tests for merge_gap parameter."""

    def test_merge_gap_closes_small_gap(self):
        # Gap of 2 between segments; merge_gap=2 should merge them
        scores = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, merge_gap=2)
        assert len(segments) == 1
        assert segments[0].start_idx == 1
        assert segments[0].end_idx == 5

    def test_merge_gap_0_does_not_merge(self):
        scores = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, merge_gap=0)
        assert len(segments) == 2

    def test_merge_gap_does_not_exceed_threshold(self):
        # Gap of 3, merge_gap=2 — should NOT merge
        scores = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, merge_gap=2)
        assert len(segments) == 2

    def test_merged_segment_peak_from_underlying_scores(self):
        scores = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5, merge_gap=1)
        assert len(segments) == 1
        # Peak should come from the merged underlying data
        assert segments[0].peak_score == pytest.approx(2.0)

    def test_invalid_merge_gap_raises(self):
        scores = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="merge_gap must be >= 0"):
            localize_anomalies(scores, threshold=0.5, merge_gap=-1)


class TestEdgeCases:
    """Edge cases: empty input, single element, 2D input."""

    def test_empty_scores_returns_empty(self):
        scores = np.array([], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        assert segments == []

    def test_single_element_above_threshold(self):
        scores = np.array([1.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        assert len(segments) == 1
        assert segments[0].start_idx == 0
        assert segments[0].end_idx == 1

    def test_single_element_below_threshold(self):
        scores = np.array([0.1], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        assert len(segments) == 0

    def test_2d_scores_raises(self):
        scores = np.ones((5, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1D"):
            localize_anomalies(scores, threshold=0.5)

    def test_list_input_accepted(self):
        scores = [0.0, 1.0, 1.0, 0.0]
        segments = localize_anomalies(scores, threshold=0.5)
        assert len(segments) == 1

    def test_result_sorted_by_start_idx(self):
        scores = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        segments = localize_anomalies(scores, threshold=0.5)
        starts = [s.start_idx for s in segments]
        assert starts == sorted(starts)
