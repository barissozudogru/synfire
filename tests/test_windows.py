import numpy as np
import pytest

from synfire.core.config import NormConfig, WindowConfig
from synfire.preprocessing.normalization import normalize_windows
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    make_shuffled_pairs,
    sliding_windows,
)


class TestSlidingWindows:
    def test_complex_series_rejected(self):
        series = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])

        with pytest.raises(ValueError, match="real numeric values"):
            sliding_windows(series, WindowConfig(window_size=2))

    @pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_series_rejected(self, invalid_value):
        series = np.array([0.0, invalid_value, 2.0])

        with pytest.raises(ValueError, match="non-finite values"):
            sliding_windows(series, WindowConfig(window_size=2))

    def test_univariate_shape(self, sine_series):
        cfg = WindowConfig(window_size=20, stride=1)
        windows = sliding_windows(sine_series, cfg)
        expected_n = (len(sine_series) - 20) // 1 + 1
        assert windows.shape == (expected_n, 20)

    def test_multivariate_shape(self, multivariate_series):
        cfg = WindowConfig(window_size=10, stride=5)
        windows = sliding_windows(multivariate_series, cfg)
        T, C = multivariate_series.shape
        expected_n = (T - 10) // 5 + 1
        assert windows.shape == (expected_n, 10 * C)

    def test_stride(self):
        series = np.arange(100, dtype=np.float64)
        cfg = WindowConfig(window_size=10, stride=10)
        windows = sliding_windows(series, cfg)
        assert windows.shape == (10, 10)
        np.testing.assert_array_equal(windows[0], np.arange(10))
        np.testing.assert_array_equal(windows[1], np.arange(10, 20))

    def test_too_short_raises(self):
        series = np.arange(5, dtype=np.float64)
        cfg = WindowConfig(window_size=10)
        with pytest.raises(ValueError, match="too short"):
            sliding_windows(series, cfg)

    @pytest.mark.parametrize("field", ["window_size", "stride"])
    def test_fractional_window_geometry_rejected(self, field):
        with pytest.raises(ValueError, match="must be an integer"):
            WindowConfig(**{field: 2.5})

    @pytest.mark.parametrize("series", [np.array(1.0), np.zeros((2, 3, 4))])
    def test_invalid_dimensions_raise_clear_error(self, series):
        with pytest.raises(ValueError, match=r"series must be 1D or 2D"):
            sliding_windows(series, WindowConfig(window_size=1))

    def test_zero_channel_series_raises_clear_error(self):
        series = np.empty((10, 0), dtype=np.float64)

        with pytest.raises(ValueError, match=r"series must contain at least one channel"):
            sliding_windows(series, WindowConfig(window_size=3))

    def test_default_config(self, sine_series):
        default_ws = WindowConfig().window_size
        windows = sliding_windows(sine_series)
        expected_n = (len(sine_series) - default_ws) // 1 + 1
        assert windows.shape == (expected_n, default_ws)

    def test_window_content_correct(self):
        series = np.arange(50, dtype=np.float64)
        cfg = WindowConfig(window_size=5, stride=1)
        windows = sliding_windows(series, cfg)
        np.testing.assert_array_equal(windows[0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(windows[3], [3, 4, 5, 6, 7])


class TestConsecutivePairs:
    def test_shapes(self, sine_series):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        left, right = make_consecutive_pairs(windows)
        assert left.shape == right.shape
        assert left.shape[0] == windows.shape[0] - 1

    def test_consecutive_overlap(self):
        series = np.arange(30, dtype=np.float64)
        windows = sliding_windows(series, WindowConfig(window_size=5, stride=1))
        left, right = make_consecutive_pairs(windows)
        np.testing.assert_array_equal(left[0], windows[0])
        np.testing.assert_array_equal(right[0], windows[1])


class TestRandomPairs:
    def test_shapes(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        left, right = make_random_pairs(windows, rng, min_gap=5)
        assert left.shape == right.shape == windows.shape

    def test_min_gap_respected(self, rng):
        series = np.arange(100, dtype=np.float64)
        windows = sliding_windows(series, WindowConfig(window_size=5, stride=1))
        left, right = make_random_pairs(windows, rng, min_gap=5)
        n = len(windows)
        for i in range(n):
            # Find which index the right window corresponds to
            for j in range(n):
                if np.array_equal(right[i], windows[j]):
                    # With circular offset, gap is min of forward and backward distance
                    gap = min(abs(i - j), n - abs(i - j))
                    assert gap >= 5, f"Gap {gap} < 5 for pair ({i}, {j})"
                    break


class TestRandomPairsEdgeCases:
    def test_zero_min_gap_still_uses_other_windows(self, rng):
        windows = np.arange(12, dtype=np.float64).reshape(4, 3)

        left, right = make_random_pairs(windows, rng, min_gap=0)

        assert not np.any(np.all(left == right, axis=1))

    def test_negative_min_gap_raises(self, rng):
        windows = np.arange(12, dtype=np.float64).reshape(4, 3)

        with pytest.raises(ValueError, match="min_gap must be >= 0"):
            make_random_pairs(windows, rng, min_gap=-1)

    @pytest.mark.parametrize("min_gap", [True, 1.5])
    def test_non_integer_min_gap_raises(self, rng, min_gap):
        windows = np.arange(12, dtype=np.float64).reshape(4, 3)

        with pytest.raises(ValueError, match="min_gap must be an integer"):
            make_random_pairs(windows, rng, min_gap=min_gap)

    def test_n_equals_1_returns_copy(self, rng):
        windows = np.array([[1.0, 2.0, 3.0]])
        left, right = make_random_pairs(windows, rng, min_gap=5)
        np.testing.assert_array_equal(left, windows)
        np.testing.assert_array_equal(right, windows)
        # Verify it's a copy, not the same object
        assert right is not left

    def test_n_less_than_2_min_gap_relaxes_constraint(self, rng):
        series = np.arange(40, dtype=np.float64)
        windows = sliding_windows(series, WindowConfig(window_size=5, stride=5))
        # 8 windows with min_gap=5 means n < 2*min_gap, gap is relaxed
        left, right = make_random_pairs(windows, rng, min_gap=5)
        assert left.shape == right.shape == windows.shape
        # Should not crash and should return valid pairs


class TestShuffledPairs:
    def test_shapes(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        left, right = make_shuffled_pairs(windows, rng)
        assert left.shape == right.shape == windows.shape

    def test_is_permutation(self, rng):
        series = np.arange(50, dtype=np.float64)
        windows = sliding_windows(series, WindowConfig(window_size=5, stride=5))
        left, right = make_shuffled_pairs(windows, rng)
        # Right should be a permutation of the same windows
        assert sorted(right[:, 0].tolist()) == sorted(left[:, 0].tolist())

    def test_shuffled_pairs_do_not_pair_a_window_with_itself(self):
        windows = np.arange(4, dtype=np.float64).reshape(2, 2)

        left, right = make_shuffled_pairs(windows, np.random.default_rng(0))

        assert not np.any(np.all(left == right, axis=1))

    def test_shuffled_pairs_with_one_window_returns_no_pairs(self):
        windows = np.array([[1.0, 2.0]])

        left, right = make_shuffled_pairs(windows, np.random.default_rng(0))

        assert left.shape == right.shape == (0, 2)


class TestNormalization:
    def test_complex_windows_rejected(self):
        windows = np.array([[1.0 + 1.0j, 2.0 + 2.0j]])

        with pytest.raises(ValueError, match=r"real numeric values"):
            normalize_windows(windows)

    @pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_windows_rejected(self, invalid_value):
        windows = np.array([[0.0, invalid_value, 2.0]])

        with pytest.raises(ValueError, match=r"non-finite values"):
            normalize_windows(windows)

    @pytest.mark.parametrize("windows", [np.arange(4.0), np.array(1.0)])
    def test_non_2d_windows_raise_clear_error(self, windows):
        with pytest.raises(ValueError, match=r"windows must be 2D"):
            normalize_windows(windows)

    def test_zscore_shape(self, sine_series):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        normed = normalize_windows(windows)
        assert normed.shape == windows.shape

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    def test_empty_feature_dimension_returns_empty_batch(self, method):
        windows = np.empty((0, 0), dtype=np.float64)

        normed = normalize_windows(windows, NormConfig(method=method))

        assert normed.shape == windows.shape
        assert normed.dtype == windows.dtype

    def test_zscore_stats(self, sine_series):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        normed = normalize_windows(windows, NormConfig(method="zscore"))
        means = np.mean(normed, axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_minmax_range(self, sine_series):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        normed = normalize_windows(windows, NormConfig(method="minmax"))
        assert np.all(normed >= -1e-10)
        assert np.all(normed <= 1.0 + 1e-10)

    def test_unknown_method_raises(self, sine_series):
        with pytest.raises(ValueError, match="method must be"):
            NormConfig(method="invalid")

    @pytest.mark.parametrize("eps", [np.nan, np.inf])
    def test_non_finite_eps_raises(self, eps):
        with pytest.raises(ValueError, match="finite"):
            NormConfig(eps=eps)

    def test_non_numeric_eps_raises_value_error(self):
        with pytest.raises(ValueError, match="eps must be finite"):
            NormConfig(eps="bad")

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64])
    def test_constant_window_with_zero_eps_returns_zeros(self, method, dtype):
        windows = np.ones((2, 4), dtype=dtype)

        normed = normalize_windows(windows, NormConfig(method=method, eps=0.0))

        np.testing.assert_array_equal(normed, np.zeros_like(windows))

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_integer_windows_match_float_normalization(self, method, dtype):
        windows = np.arange(8, dtype=dtype).reshape(2, 4)
        config = NormConfig(method=method, eps=0.0)

        normed = normalize_windows(windows, config)
        expected = normalize_windows(windows.astype(np.float64), config)

        np.testing.assert_allclose(normed, expected)
        assert np.issubdtype(normed.dtype, np.floating)

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_preserves_float_input_dtype(self, method, dtype):
        windows = np.arange(8, dtype=dtype).reshape(2, 4)

        normed = normalize_windows(windows, NormConfig(method=method))

        assert normed.dtype == windows.dtype

    def test_large_float32_windows_do_not_overflow_during_zscore(self):
        windows = np.array([[1e20, -1e20]], dtype=np.float32)

        normed = normalize_windows(windows, NormConfig(method="zscore"))

        np.testing.assert_allclose(normed, [[1.0, -1.0]], rtol=1e-6)
