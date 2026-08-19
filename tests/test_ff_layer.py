import numpy as np
import pytest

from synfire.core.config import FFLayerConfig, WindowConfig
from synfire.layers.ff_layer import (
    _pairwise_sq_distances,
    compute_loss,
    forward,
    goodness,
    init_layer,
    train_layer,
    train_step,
)
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    sliding_windows,
)


class TestFFLayerInit:
    def test_shapes(self):
        cfg = FFLayerConfig(input_dim=20, hidden_dim=32)
        state = init_layer(cfg)
        assert state.W.shape == (32, 20)
        assert state.b.shape == (32,)

    def test_deterministic(self):
        cfg = FFLayerConfig(input_dim=20, hidden_dim=32, seed=123)
        s1 = init_layer(cfg)
        s2 = init_layer(cfg)
        np.testing.assert_array_equal(s1.W, s2.W)


class TestForwardAndGoodness:
    def test_forward_shape(self):
        cfg = FFLayerConfig(input_dim=20, hidden_dim=32)
        state = init_layer(cfg)
        x = np.random.default_rng(0).standard_normal((10, 20))
        h = forward(state, x)
        assert h.shape == (10, 32)

    def test_forward_nonneg(self):
        cfg = FFLayerConfig(input_dim=20, hidden_dim=32)
        state = init_layer(cfg)
        x = np.random.default_rng(0).standard_normal((10, 20))
        h = forward(state, x)
        assert np.all(h >= 0)

    def test_goodness_shape(self):
        h = np.random.default_rng(0).standard_normal((10, 32))
        h = np.maximum(h, 0)
        g = goodness(h)
        assert g.shape == (10,)

    def test_goodness_nonneg(self):
        h = np.abs(np.random.default_rng(0).standard_normal((10, 32)))
        g = goodness(h)
        assert np.all(g >= 0)


class TestComputeLoss:
    def test_loss_finite(self):
        g_pos = np.array([3.0, 4.0, 3.5])
        g_neg = np.array([1.0, 0.5, 0.8])
        loss, d_pos, d_neg = compute_loss(g_pos, g_neg, threshold=2.0)
        assert np.isfinite(loss)
        assert d_pos.shape == (3,)
        assert d_neg.shape == (3,)

    def test_loss_decreases_with_separation(self):
        # Well-separated: pos >> threshold >> neg -> lower loss
        g_pos_good = np.array([5.0, 6.0])
        g_neg_good = np.array([0.1, 0.2])
        loss_good, _, _ = compute_loss(g_pos_good, g_neg_good, threshold=2.0)

        g_pos_bad = np.array([2.1, 2.0])
        g_neg_bad = np.array([1.9, 2.0])
        loss_bad, _, _ = compute_loss(g_pos_bad, g_neg_bad, threshold=2.0)

        assert loss_good < loss_bad


class TestTrainStep:
    def test_loss_returned(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)

        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFLayerConfig(input_dim=40, hidden_dim=32, lr=0.01, threshold=2.0)
        state = init_layer(cfg)
        new_state, loss = train_step(state, x_pos, x_neg)
        assert np.isfinite(loss)
        assert new_state.W.shape == state.W.shape


class TestGradientCorrectness:
    """Finite-difference gradient checks for the FF layer."""

    def test_weight_gradient(self):
        rng = np.random.default_rng(99)
        cfg = FFLayerConfig(input_dim=4, hidden_dim=6, lr=1.0, threshold=2.0, epochs=1, seed=99)
        state = init_layer(cfg)
        x_pos = rng.standard_normal((8, 4))
        x_neg = rng.standard_normal((8, 4))

        # lr=1 trick: dW = W_old - W_new when lr=1
        W_old = state.W.copy()
        new_state, _ = train_step(state, x_pos, x_neg)
        analytic_dW = W_old - new_state.W

        # Numerical gradient via central differences
        eps = 1e-5
        numerical_dW = np.zeros_like(state.W)
        for i in range(state.W.shape[0]):
            for j in range(state.W.shape[1]):
                W_plus = state.W.copy()
                W_plus[i, j] += eps
                h_pos_p = np.maximum(x_pos @ W_plus.T + state.b, 0)
                g_pos_p = np.mean(h_pos_p**2, axis=1)
                h_neg_p = np.maximum(x_neg @ W_plus.T + state.b, 0)
                g_neg_p = np.mean(h_neg_p**2, axis=1)
                loss_p, _, _ = compute_loss(g_pos_p, g_neg_p, cfg.threshold)

                W_minus = state.W.copy()
                W_minus[i, j] -= eps
                h_pos_m = np.maximum(x_pos @ W_minus.T + state.b, 0)
                g_pos_m = np.mean(h_pos_m**2, axis=1)
                h_neg_m = np.maximum(x_neg @ W_minus.T + state.b, 0)
                g_neg_m = np.mean(h_neg_m**2, axis=1)
                loss_m, _, _ = compute_loss(g_pos_m, g_neg_m, cfg.threshold)

                numerical_dW[i, j] = (loss_p - loss_m) / (2 * eps)

        np.testing.assert_allclose(analytic_dW, numerical_dW, atol=1e-4, rtol=1e-3)

    def test_bias_gradient(self):
        rng = np.random.default_rng(77)
        cfg = FFLayerConfig(input_dim=4, hidden_dim=6, lr=1.0, threshold=2.0, epochs=1, seed=77)
        state = init_layer(cfg)
        x_pos = rng.standard_normal((8, 4))
        x_neg = rng.standard_normal((8, 4))

        b_old = state.b.copy()
        new_state, _ = train_step(state, x_pos, x_neg)
        analytic_db = b_old - new_state.b

        eps = 1e-5
        numerical_db = np.zeros_like(state.b)
        for i in range(state.b.shape[0]):
            b_plus = state.b.copy()
            b_plus[i] += eps
            h_pos_p = np.maximum(x_pos @ state.W.T + b_plus, 0)
            g_pos_p = np.mean(h_pos_p**2, axis=1)
            h_neg_p = np.maximum(x_neg @ state.W.T + b_plus, 0)
            g_neg_p = np.mean(h_neg_p**2, axis=1)
            loss_p, _, _ = compute_loss(g_pos_p, g_neg_p, cfg.threshold)

            b_minus = state.b.copy()
            b_minus[i] -= eps
            h_pos_m = np.maximum(x_pos @ state.W.T + b_minus, 0)
            g_pos_m = np.mean(h_pos_m**2, axis=1)
            h_neg_m = np.maximum(x_neg @ state.W.T + b_minus, 0)
            g_neg_m = np.mean(h_neg_m**2, axis=1)
            loss_m, _, _ = compute_loss(g_pos_m, g_neg_m, cfg.threshold)

            numerical_db[i] = (loss_p - loss_m) / (2 * eps)

        np.testing.assert_allclose(analytic_db, numerical_db, atol=1e-4, rtol=1e-3)


class TestTrainLayer:
    def test_loss_decreases(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)

        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFLayerConfig(
            input_dim=40, hidden_dim=32, lr=0.01, threshold=2.0, epochs=50
        )
        state = init_layer(cfg)
        trained, losses = train_layer(state, x_pos, x_neg)

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_goodness_separation(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)

        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFLayerConfig(
            input_dim=40, hidden_dim=32, lr=0.01, threshold=2.0, epochs=100
        )
        state = init_layer(cfg)
        trained, _ = train_layer(state, x_pos, x_neg)

        h_pos = forward(trained, x_pos)
        h_neg = forward(trained, x_neg)
        g_pos = goodness(h_pos)
        g_neg = goodness(h_neg)

        # After training, positive goodness should be higher on average
        assert np.mean(g_pos) > np.mean(g_neg)


class TestPairwiseSqDistances:
    """Tests for the chunked pairwise squared-distance helper (H-1 fix)."""

    def test_matches_broadcast_small(self):
        """Chunked result matches the naive broadcast result for small inputs."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((20, 8))
        b = rng.standard_normal((15, 8))
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        expected = np.sum(diff ** 2, axis=2)
        got = _pairwise_sq_distances(a, b)
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_chunked_path_matches_broadcast(self):
        """Force the chunked path by setting a tiny element limit and verify output."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((30, 6))
        b = rng.standard_normal((25, 6))
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        expected = np.sum(diff ** 2, axis=2)
        # Patch the limit to force chunked path even for this small input.
        import synfire.layers.ff_layer as ff_mod
        original = ff_mod._HARD_NEG_ELEMENT_LIMIT
        try:
            ff_mod._HARD_NEG_ELEMENT_LIMIT = 1  # force chunked for any input
            got = _pairwise_sq_distances(a, b, chunk_size=8)
        finally:
            ff_mod._HARD_NEG_ELEMENT_LIMIT = original
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_output_non_negative(self):
        """Squared distances are always non-negative (clamp guards numerical noise)."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((10, 4))
        b = a + rng.standard_normal((10, 4)) * 1e-8  # near-identical rows
        dists = _pairwise_sq_distances(a, b)
        assert np.all(dists >= 0.0)

    def test_hard_strategy_uses_distances(self):
        """With negative_strategy='hard', the returned negatives are the closest ones."""
        rng = np.random.default_rng(3)
        D = 4
        x_pos = rng.standard_normal((5, D))
        x_neg = rng.standard_normal((10, D))
        cfg = FFLayerConfig(input_dim=D, hidden_dim=8, negative_strategy="hard")
        state = init_layer(cfg)

        pytest.importorskip("synfire.layers.ff_layer")
        from synfire.layers.ff_layer import _mine_hard_negatives
        result = _mine_hard_negatives(x_pos, x_neg, state, epoch=0, total_epochs=1)

        # Verify each selected negative is the closest one in x_neg.
        dists = _pairwise_sq_distances(x_pos, x_neg)
        expected_indices = np.argmin(dists, axis=1)
        np.testing.assert_array_equal(result, x_neg[expected_indices])
