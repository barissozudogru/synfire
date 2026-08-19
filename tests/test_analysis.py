"""Tests for synfire.analysis: layer decomposition, activation stats, prototype utilization."""

from __future__ import annotations

import numpy as np
import pytest

from synfire import SynfireConfig, SynfirePipeline
from synfire.analysis import (
    LayerStats,
    activation_statistics,
    layer_anomaly_decomposition,
    prototype_utilization,
)
from synfire.core.config import FFStackConfig, HebbianConfig, WindowConfig


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Small fitted pipeline shared across analysis tests."""
    t = np.arange(500, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)

    config = SynfireConfig(
        window=WindowConfig(window_size=10, stride=1),
        ff_stack=FFStackConfig(layer_dims=(16, 8), lr=0.01, threshold=2.0, epochs_per_layer=5),
        hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=5),
    )
    pipeline = SynfirePipeline(config)
    pipeline.fit(series)
    return pipeline


@pytest.fixture(scope="module")
def test_pairs(fitted_pipeline):
    """A small batch of test input pairs derived from the fitted pipeline's series."""
    t = np.arange(50, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)
    return fitted_pipeline._prepare_test_pairs(series)


class TestLayerAnomalyDecomposition:
    """Tests for layer_anomaly_decomposition."""

    def test_returns_2d_array(self, fitted_pipeline, test_pairs):
        result = layer_anomaly_decomposition(fitted_pipeline._stack, test_pairs)
        assert result.ndim == 2

    def test_shape_batch_x_nlayers(self, fitted_pipeline, test_pairs):
        result = layer_anomaly_decomposition(fitted_pipeline._stack, test_pairs)
        n_layers = len(fitted_pipeline._stack.layers)
        assert result.shape == (len(test_pairs), n_layers)

    def test_custom_threshold_shifts_deficits(self, fitted_pipeline, test_pairs):
        result_default = layer_anomaly_decomposition(fitted_pipeline._stack, test_pairs)
        result_high = layer_anomaly_decomposition(
            fitted_pipeline._stack, test_pairs, threshold=100.0
        )
        # With a very high threshold, all deficits should be larger
        assert np.all(result_high >= result_default - 1e-9)

    def test_values_are_finite(self, fitted_pipeline, test_pairs):
        result = layer_anomaly_decomposition(fitted_pipeline._stack, test_pairs)
        assert np.all(np.isfinite(result))

    def test_none_threshold_uses_first_layer_config(self, fitted_pipeline, test_pairs):
        expected_threshold = fitted_pipeline._stack.layers[0].config.threshold
        result_none = layer_anomaly_decomposition(
            fitted_pipeline._stack, test_pairs, threshold=None
        )
        result_explicit = layer_anomaly_decomposition(
            fitted_pipeline._stack, test_pairs, threshold=expected_threshold
        )
        np.testing.assert_allclose(result_none, result_explicit)


class TestActivationStatistics:
    """Tests for activation_statistics."""

    def test_returns_list_of_layer_stats(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        assert isinstance(stats, list)
        assert all(isinstance(s, LayerStats) for s in stats)

    def test_one_stat_per_layer(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        n_layers = len(fitted_pipeline._stack.layers)
        assert len(stats) == n_layers

    def test_layer_indices_sequential(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        for i, stat in enumerate(stats):
            assert stat.layer_index == i

    def test_expected_keys_present(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        required_attrs = {
            "layer_index", "mean_activation", "std_activation",
            "sparsity", "mean_goodness", "std_goodness",
        }
        for stat in stats:
            for attr in required_attrs:
                assert hasattr(stat, attr), f"LayerStats missing attribute: {attr}"

    def test_sparsity_in_valid_range(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        for stat in stats:
            assert 0.0 <= stat.sparsity <= 1.0

    def test_std_activation_non_negative(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        for stat in stats:
            assert stat.std_activation >= 0.0

    def test_values_are_finite(self, fitted_pipeline, test_pairs):
        stats = activation_statistics(fitted_pipeline._stack, test_pairs)
        for stat in stats:
            assert np.isfinite(stat.mean_activation)
            assert np.isfinite(stat.std_activation)
            assert np.isfinite(stat.sparsity)
            assert np.isfinite(stat.mean_goodness)
            assert np.isfinite(stat.std_goodness)


class TestPrototypeUtilization:
    """Tests for prototype_utilization."""

    def test_returns_integer_counts(self, fitted_pipeline, test_pairs):
        reps = fitted_pipeline.transform(
            np.sin(2 * np.pi * np.arange(50, dtype=np.float64) / 50)
        )
        counts = prototype_utilization(fitted_pipeline._hebbian, reps)
        assert counts.dtype in (np.int32, np.int64, np.intp)

    def test_shape_equals_n_prototypes(self, fitted_pipeline):
        reps = fitted_pipeline.transform(
            np.sin(2 * np.pi * np.arange(50, dtype=np.float64) / 50)
        )
        n_proto = fitted_pipeline.config.hebbian.n_prototypes
        counts = prototype_utilization(fitted_pipeline._hebbian, reps)
        assert counts.shape == (n_proto,)

    def test_counts_sum_to_n_samples(self, fitted_pipeline):
        series = np.sin(2 * np.pi * np.arange(50, dtype=np.float64) / 50)
        reps = fitted_pipeline.transform(series)
        counts = prototype_utilization(fitted_pipeline._hebbian, reps)
        assert counts.sum() == len(reps)

    def test_counts_non_negative(self, fitted_pipeline):
        reps = fitted_pipeline.transform(
            np.sin(2 * np.pi * np.arange(50, dtype=np.float64) / 50)
        )
        counts = prototype_utilization(fitted_pipeline._hebbian, reps)
        assert np.all(counts >= 0)

    def test_counts_with_larger_batch(self, fitted_pipeline):
        series = np.sin(2 * np.pi * np.arange(300, dtype=np.float64) / 50)
        reps = fitted_pipeline.transform(series)
        n_proto = fitted_pipeline.config.hebbian.n_prototypes
        counts = prototype_utilization(fitted_pipeline._hebbian, reps)
        assert counts.shape == (n_proto,)
        assert counts.sum() == len(reps)

    def test_all_prototypes_have_valid_counts(self, fitted_pipeline):
        """Every prototype count must be a non-negative integer."""
        series = np.sin(2 * np.pi * np.arange(300, dtype=np.float64) / 50)
        reps = fitted_pipeline.transform(series)
        counts = prototype_utilization(fitted_pipeline._hebbian, reps)
        assert np.all(counts >= 0)
        assert counts.dtype.kind == "i"  # integer dtype
