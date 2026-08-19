"""Mapping score indices back to sample indices.

anomaly_scores returns one value per window transition, so it is shorter than
the input and offset from it. Without an explicit mapping, indexing the input
with a score index is off by the window offset, which matters because locating
the anomaly in time is the point of an anomaly detector.
"""

import numpy as np
import pytest

from synfire import SynfirePipeline


@pytest.fixture(scope="module")
def fitted():
    t = np.arange(600, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)
    pipeline = SynfirePipeline()
    pipeline.fit(series)
    return pipeline


def test_score_length_follows_window_geometry(fitted):
    t = np.arange(300, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)
    scores = fitted.anomaly_scores(series)

    w = fitted.config.window
    n_windows = (len(series) - w.window_size) // w.stride + 1
    assert len(scores) == n_windows - 1


def test_score_index_maps_into_the_series(fitted):
    t = np.arange(300, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)
    scores = fitted.anomaly_scores(series)

    for idx in (0, len(scores) // 2, len(scores) - 1):
        sample = fitted.score_index_to_sample(idx)
        assert 0 <= sample < len(series)


def test_window_bounds_are_within_the_series(fitted):
    t = np.arange(300, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)
    scores = fitted.anomaly_scores(series)

    start, end = fitted.score_window_bounds(len(scores) - 1)
    assert start < end
    assert end <= len(series)


def test_mapping_is_monotonic(fitted):
    samples = [fitted.score_index_to_sample(i) for i in range(10)]
    assert samples == sorted(samples)
    assert len(set(samples)) == len(samples)


def test_naive_indexing_is_offset(fitted):
    # Documents why the mapping exists: a score index is not a sample index.
    w = fitted.config.window
    assert fitted.score_index_to_sample(0) == w.stride
    assert fitted.score_index_to_sample(0) != 0
