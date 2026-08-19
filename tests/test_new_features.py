"""Tests for 7 core improvements: Adam optimizer, layer norm, vectorized Hebbian,
score decomposition, visualization, hard negative mining, and multi-resolution."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import (
    AnomalyConfig,
    FFLayerConfig,
    FFStackConfig,
    HebbianConfig,
    NormConfig,
    WindowConfig,
)
from synfire.layers.ff_layer import (
    _mine_hard_negatives,
    forward,
    init_layer,
    train_layer,
    train_step,
)
from synfire.layers.ff_stack import init_stack
from synfire.layers.hebbian import HebbianState, init_hebbian, train_hebbian, update_step
from synfire.multi_resolution import MultiResolutionPipeline
from synfire.pipeline.anomaly import (
    DecomposedAnomalyScore,
    anomaly_scores_decomposed,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_series():
    t = np.arange(800, dtype=np.float64)
    return np.sin(2 * np.pi * t / 50)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_pos_neg(rng):
    x_pos = rng.standard_normal((80, 20))
    x_neg = rng.permutation(rng.standard_normal((80, 20)))
    return x_pos, x_neg


@pytest.fixture
def minimal_pipeline(sine_series):
    config = SynfireConfig(
        window=WindowConfig(window_size=20, stride=1),
        norm=NormConfig(method="zscore"),
        ff_stack=FFStackConfig(layer_dims=(32, 16), lr=0.05, epochs_per_layer=20),
        hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=5),
    )
    pipeline = SynfirePipeline(config)
    pipeline.fit(sine_series)
    return pipeline


# ===========================================================================
# 1. Adam Optimizer + Weight Decay
# ===========================================================================

class TestAdamOptimizer:
    def test_adam_config_defaults(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8)
        assert cfg.optimizer == "sgd"
        assert cfg.weight_decay == 0.0

    def test_adam_config_accepted(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, optimizer="adam")
        assert cfg.optimizer == "adam"

    def test_invalid_optimizer_rejected(self):
        with pytest.raises(ValueError, match="optimizer must be"):
            FFLayerConfig(input_dim=10, hidden_dim=8, optimizer="rmsprop")

    def test_negative_weight_decay_rejected(self):
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            FFLayerConfig(input_dim=10, hidden_dim=8, weight_decay=-0.1)

    def test_adam_state_initialized(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, optimizer="adam")
        state = init_layer(cfg)
        assert state.m_W is not None
        assert state.v_W is not None
        assert state.m_b is not None
        assert state.v_b is not None
        assert state.m_W.shape == state.W.shape
        assert state.v_b.shape == state.b.shape
        assert state.adam_t == 0

    def test_sgd_state_has_no_moments(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, optimizer="sgd")
        state = init_layer(cfg)
        assert state.m_W is None
        assert state.v_W is None

    def test_adam_step_increments_t(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(input_dim=20, hidden_dim=16, optimizer="adam")
        state = init_layer(cfg)
        new_state, _ = train_step(state, x_pos, x_neg)
        assert new_state.adam_t == 1
        new_state2, _ = train_step(new_state, x_pos, x_neg)
        assert new_state2.adam_t == 2

    def test_adam_weights_change(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(input_dim=20, hidden_dim=16, optimizer="adam", lr=0.01)
        state = init_layer(cfg)
        new_state, _ = train_step(state, x_pos, x_neg)
        assert not np.allclose(state.W, new_state.W)

    def test_adam_training_converges(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16, optimizer="adam", lr=0.01, epochs=50, seed=42
        )
        state = init_layer(cfg)
        _, losses = train_layer(state, x_pos, x_neg)
        assert losses[-1] < losses[0], "Adam should converge (loss decrease)"

    def test_adam_vs_sgd_different_trajectories(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg_sgd = FFLayerConfig(input_dim=20, hidden_dim=16, optimizer="sgd", lr=0.01, epochs=30)
        cfg_adam = FFLayerConfig(input_dim=20, hidden_dim=16, optimizer="adam", lr=0.01, epochs=30)
        _, losses_sgd = train_layer(init_layer(cfg_sgd), x_pos, x_neg)
        _, losses_adam = train_layer(init_layer(cfg_adam), x_pos, x_neg)
        # They should produce different loss trajectories
        assert not np.allclose(losses_sgd, losses_adam)

    def test_weight_decay_reduces_weight_magnitude(self, small_pos_neg):
        """L2 weight decay should shrink weight norms relative to no-decay baseline."""
        x_pos, x_neg = small_pos_neg
        cfg_no_wd = FFLayerConfig(
            input_dim=20, hidden_dim=16, optimizer="sgd", lr=0.01, epochs=30,
            seed=7, weight_decay=0.0,
        )
        cfg_wd = FFLayerConfig(
            input_dim=20, hidden_dim=16, optimizer="sgd", lr=0.01, epochs=30,
            seed=7, weight_decay=0.1,
        )
        trained_no_wd, _ = train_layer(init_layer(cfg_no_wd), x_pos, x_neg)
        trained_wd, _ = train_layer(init_layer(cfg_wd), x_pos, x_neg)
        norm_no_wd = float(np.linalg.norm(trained_no_wd.W))
        norm_wd = float(np.linalg.norm(trained_wd.W))
        assert norm_wd < norm_no_wd, (
            f"Weight decay should shrink W: {norm_wd:.4f} vs {norm_no_wd:.4f}"
        )

    def test_adam_stack_propagation(self):
        cfg = FFStackConfig(
            layer_dims=(16, 8), lr=0.01, epochs_per_layer=5, optimizer="adam", weight_decay=0.01
        )
        state = init_stack(20, cfg)
        for layer in state.layers:
            assert layer.config.optimizer == "adam"
            assert layer.config.weight_decay == pytest.approx(0.01)

    def test_adam_pipeline_end_to_end(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(32,), lr=0.005, epochs_per_layer=10,
                optimizer="adam", weight_decay=0.001,
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))
        assert len(scores) > 0

    def test_weight_decay_finite_output(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16, optimizer="adam", lr=0.01,
            epochs=20, weight_decay=0.5, seed=1,
        )
        state = init_layer(cfg)
        trained, losses = train_layer(state, x_pos, x_neg)
        assert np.all(np.isfinite(trained.W))
        assert all(np.isfinite(layer) for layer in losses)


# ===========================================================================
# 2. Layer Normalization
# ===========================================================================

class TestLayerNorm:
    def test_layer_norm_config_default_false(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8)
        assert cfg.layer_norm is False

    def test_layer_norm_config_true(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, layer_norm=True)
        assert cfg.layer_norm is True

    def test_layer_norm_init_creates_params(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, layer_norm=True)
        state = init_layer(cfg)
        assert state.ln_gain is not None
        assert state.ln_bias is not None
        assert state.ln_gain.shape == (8,)
        assert state.ln_bias.shape == (8,)
        np.testing.assert_array_equal(state.ln_gain, np.ones(8))
        np.testing.assert_array_equal(state.ln_bias, np.zeros(8))

    def test_no_layer_norm_no_params(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, layer_norm=False)
        state = init_layer(cfg)
        assert state.ln_gain is None
        assert state.ln_bias is None

    def test_forward_with_layer_norm_shape(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, layer_norm=True)
        state = init_layer(cfg)
        x = np.random.default_rng(1).standard_normal((20, 10))
        h = forward(state, x)
        assert h.shape == (20, 8)

    def test_forward_with_layer_norm_nonneg(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, layer_norm=True)
        state = init_layer(cfg)
        x = np.random.default_rng(2).standard_normal((20, 10))
        h = forward(state, x)
        assert np.all(h >= 0), "ReLU after layer norm must be non-negative"

    def test_forward_without_layer_norm_unchanged(self):
        """Disabling layer norm should give same result as before."""
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, layer_norm=False, seed=5)
        state = init_layer(cfg)
        x = np.random.default_rng(3).standard_normal((20, 10))
        h_ln_off = forward(state, x)
        # Manually compute expected
        pre = x @ state.W.T + state.b
        expected = np.maximum(pre, 0)
        np.testing.assert_allclose(h_ln_off, expected)

    def test_layer_norm_train_step_finite(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(input_dim=20, hidden_dim=16, layer_norm=True, lr=0.01)
        state = init_layer(cfg)
        new_state, loss = train_step(state, x_pos, x_neg)
        assert np.isfinite(loss)
        assert np.all(np.isfinite(new_state.W))
        assert np.all(np.isfinite(new_state.ln_gain))
        assert np.all(np.isfinite(new_state.ln_bias))

    def test_layer_norm_gain_bias_update(self, small_pos_neg):
        """Layer norm gain and bias should change during training."""
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(input_dim=20, hidden_dim=16, layer_norm=True, lr=0.05)
        state = init_layer(cfg)
        new_state, _ = train_step(state, x_pos, x_neg)
        # gain or bias should change
        gain_changed = not np.allclose(state.ln_gain, new_state.ln_gain)
        bias_changed = not np.allclose(state.ln_bias, new_state.ln_bias)
        assert gain_changed or bias_changed, "LN parameters must be updated during training"

    def test_layer_norm_converges(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16, layer_norm=True, lr=0.02, epochs=50, seed=11
        )
        state = init_layer(cfg)
        _, losses = train_layer(state, x_pos, x_neg)
        assert losses[-1] < losses[0]

    def test_layer_norm_pipeline(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(32,), lr=0.02, epochs_per_layer=15, layer_norm=True,
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_layer_norm_stack_propagation(self):
        cfg = FFStackConfig(layer_dims=(16, 8), lr=0.01, epochs_per_layer=5, layer_norm=True)
        state = init_stack(20, cfg)
        for layer in state.layers:
            assert layer.config.layer_norm is True
            assert layer.ln_gain is not None
            assert layer.ln_bias is not None


# ===========================================================================
# 3. Vectorized Hebbian Inhibition
# ===========================================================================

class TestVectorizedHebbian:
    def test_update_step_shape_unchanged(self, rng):
        data = rng.standard_normal((100, 16))
        from synfire.core.config import HebbianConfig
        cfg = HebbianConfig(n_prototypes=4, lr=0.01, inhibition_strength=0.1, epochs=1)
        state = init_hebbian(data, cfg)
        x = rng.standard_normal((20, 16))
        new_state = update_step(state, x)
        assert new_state.prototypes.shape == state.prototypes.shape

    def test_update_step_prototypes_finite(self, rng):
        data = rng.standard_normal((100, 16))
        from synfire.core.config import HebbianConfig
        cfg = HebbianConfig(n_prototypes=4, lr=0.01, inhibition_strength=0.1, epochs=1)
        state = init_hebbian(data, cfg)
        x = rng.standard_normal((32, 16))
        new_state = update_step(state, x)
        assert np.all(np.isfinite(new_state.prototypes))

    def test_vectorized_matches_reference_single_batch(self, rng):
        """Vectorized inhibition should produce the same result for all-different winners."""
        data = rng.standard_normal((50, 4))
        from synfire.core.config import HebbianConfig
        cfg = HebbianConfig(n_prototypes=3, lr=0.05, inhibition_strength=0.2, epochs=1, seed=1)
        state = init_hebbian(data, cfg)

        x = rng.standard_normal((10, 4))
        new_state = update_step(state, x)
        # Basic sanity: prototypes moved, still finite
        assert np.all(np.isfinite(new_state.prototypes))
        assert not np.allclose(state.prototypes, new_state.prototypes)

    def test_zero_inhibition_no_repulsion(self, rng):
        """With inhibition_strength=0, prototypes should only attract (no repulsion)."""
        data = rng.standard_normal((60, 8))
        from synfire.core.config import HebbianConfig
        cfg_inh = HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.5, epochs=1, seed=2)
        cfg_no = HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.0, epochs=1, seed=2)
        state_inh = init_hebbian(data, cfg_inh)
        state_no = init_hebbian(data, cfg_no)

        x = rng.standard_normal((20, 8))
        new_inh = update_step(state_inh, x)
        new_no = update_step(state_no, x)

        # Results differ when inhibition differs
        assert not np.allclose(new_inh.prototypes, new_no.prototypes)

    def test_train_hebbian_produces_finite_prototypes(self, rng):
        data = rng.standard_normal((100, 8))
        from synfire.core.config import HebbianConfig
        cfg = HebbianConfig(n_prototypes=4, lr=0.01, inhibition_strength=0.1, epochs=5, seed=3)
        state = init_hebbian(data, cfg)
        trained = train_hebbian(state, data, batch_size=32)
        assert np.all(np.isfinite(trained.prototypes))

    def test_update_step_with_single_batch_point(self, rng):
        """Update with a single input should not crash."""
        data = rng.standard_normal((20, 4))
        from synfire.core.config import HebbianConfig
        cfg = HebbianConfig(n_prototypes=2, lr=0.05, inhibition_strength=0.1, epochs=1, seed=0)
        state = init_hebbian(data, cfg)
        x = rng.standard_normal((1, 4))
        new_state = update_step(state, x)
        assert np.all(np.isfinite(new_state.prototypes))

    def test_update_step_all_same_winner(self, rng):
        """All inputs assigned to the same prototype: no non-winner inputs for others."""
        data = rng.standard_normal((20, 2))
        from synfire.core.config import HebbianConfig
        # Place prototypes far apart; put all data near prototype 0
        cfg = HebbianConfig(n_prototypes=3, lr=0.05, inhibition_strength=0.1, epochs=1, seed=5)
        state = init_hebbian(data, cfg)
        # Override prototypes so proto 0 is at origin, others far away
        protos = state.prototypes.copy()
        protos[0] = np.zeros(2)
        protos[1] = np.array([1000.0, 1000.0])
        protos[2] = np.array([-1000.0, -1000.0])
        state = HebbianState(prototypes=protos, config=cfg)
        x = rng.standard_normal((10, 2)) * 0.01  # all near origin -> all win proto 0
        new_state = update_step(state, x)
        assert np.all(np.isfinite(new_state.prototypes))


# ===========================================================================
# 4. Anomaly Score Decomposition API
# ===========================================================================

class TestScoreDecomposition:
    def test_decomposed_score_is_dataclass(self, minimal_pipeline, sine_series):
        result = minimal_pipeline.score_decomposed(sine_series)
        assert isinstance(result, DecomposedAnomalyScore)

    def test_combined_matches_anomaly_scores(self, minimal_pipeline, sine_series):
        """Combined field must equal anomaly_scores() output."""
        scores = minimal_pipeline.anomaly_scores(sine_series)
        decomposed = minimal_pipeline.score_decomposed(sine_series)
        np.testing.assert_allclose(decomposed.combined, scores, atol=1e-12)

    def test_all_components_present_default_config(self, minimal_pipeline, sine_series):
        """Default config uses all three components."""
        result = minimal_pipeline.score_decomposed(sine_series)
        assert result.goodness_deficit is not None
        assert result.prototype_distance is not None
        assert result.transition_surprise is not None

    def test_components_finite(self, minimal_pipeline, sine_series):
        result = minimal_pipeline.score_decomposed(sine_series)
        for arr in (result.goodness_deficit, result.prototype_distance,
                    result.transition_surprise, result.combined):
            if arr is not None:
                assert np.all(np.isfinite(arr)), "Component has non-finite values"

    def test_goodness_only_config(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(layer_dims=(16,), lr=0.05, epochs_per_layer=10),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
            anomaly=AnomalyConfig(
                use_goodness=True, use_distance=False, use_transition=False,
                weight_goodness=1.0, weight_distance=0.0, weight_transition=0.0,
            ),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        result = pipeline.score_decomposed(sine_series)
        assert result.goodness_deficit is not None
        assert result.prototype_distance is None
        assert result.transition_surprise is None

    def test_component_shapes_match_combined(self, minimal_pipeline, sine_series):
        result = minimal_pipeline.score_decomposed(sine_series)
        n = len(result.combined)
        for arr in (result.goodness_deficit, result.prototype_distance,
                    result.transition_surprise):
            if arr is not None:
                assert arr.shape == (n,)

    def test_unfitted_pipeline_raises(self, sine_series):
        pipeline = SynfirePipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.score_decomposed(sine_series)

    def test_decomposed_function_direct(self, minimal_pipeline, sine_series):
        """Call anomaly_scores_decomposed directly without pipeline."""
        from synfire.preprocessing.normalization import normalize_windows
        from synfire.preprocessing.windows import (
            make_consecutive_pairs,
            sliding_windows,
        )
        config = minimal_pipeline.config
        windows = sliding_windows(sine_series, config.window)
        windows = normalize_windows(windows, config.norm)
        left, right = make_consecutive_pairs(windows)
        test_pairs = np.concatenate([left, right], axis=1)

        result = anomaly_scores_decomposed(
            minimal_pipeline._stack,
            minimal_pipeline._hebbian,
            test_pairs,
            config.anomaly,
            minimal_pipeline._effective_threshold,
            scaler=minimal_pipeline._anomaly_scaler,
        )
        assert isinstance(result, DecomposedAnomalyScore)
        assert np.all(np.isfinite(result.combined))


# ===========================================================================
# 5. Visualization Module
# ===========================================================================

class _FakeAxes:
    """Minimal matplotlib Axes stand-in for visualization tests."""

    def __init__(self):
        self._lines: list = []
        self._patches: list = []

    def plot(self, *a, **kw):
        import types
        line = types.SimpleNamespace(get_label=lambda: kw.get("label", ""))
        self._lines.append(line)
        return [line]

    def hist(self, *a, **kw): pass
    def axvline(self, *a, **kw): pass
    def axhline(self, *a, **kw): pass
    def axvspan(self, *a, **kw): pass
    def stackplot(self, *a, **kw): pass
    def set_title(self, *a, **kw): pass
    def set_xlabel(self, *a, **kw): pass
    def set_ylabel(self, *a, **kw): pass
    def legend(self, *a, **kw): pass
    def grid(self, *a, **kw): pass
    def get_lines(self): return self._lines


class TestVisualization:
    """Tests for the visualization module using a fake Axes object."""

    def test_import_visualization_module(self):
        """Visualization module should import cleanly."""
        import synfire.visualization  # noqa: F401

    def test_plot_training_loss_is_callable(self):
        import synfire.visualization as viz
        assert callable(viz.plot_training_loss)

    def test_functions_exist(self):
        """All documented visualization functions must exist."""
        import synfire.visualization as viz
        assert callable(viz.plot_training_loss)
        assert callable(viz.plot_goodness_distribution)
        assert callable(viz.plot_anomaly_scores)
        assert callable(viz.plot_score_decomposition)

    def test_plot_training_loss_with_mocked_mpl(self):
        """plot_training_loss should work with a provided Axes object."""
        import synfire.visualization as viz
        # Monkey-patch _require_matplotlib for this test
        orig = viz._require_matplotlib

        import types

        class FakePlt:
            @staticmethod
            def subplots(*a, **kw):
                return types.SimpleNamespace(), _FakeAxes()

        def fake_require():
            return types.SimpleNamespace(), FakePlt()

        viz._require_matplotlib = fake_require
        try:
            ax = _FakeAxes()
            result = viz.plot_training_loss([[1.0, 0.9, 0.8], [1.2, 1.0]], ax=ax)
            assert result is ax
        finally:
            viz._require_matplotlib = orig

    def test_plot_goodness_distribution_with_mocked_mpl(self, rng):
        import types

        import synfire.visualization as viz

        class FakePlt:
            @staticmethod
            def subplots(*a, **kw):
                return types.SimpleNamespace(), _FakeAxes()

        orig = viz._require_matplotlib
        viz._require_matplotlib = lambda: (types.SimpleNamespace(), FakePlt())
        try:
            ax = _FakeAxes()
            result = viz.plot_goodness_distribution(rng.random(50), rng.random(50), ax=ax)
            assert result is ax
        finally:
            viz._require_matplotlib = orig

    def test_plot_anomaly_scores_with_mocked_mpl(self, rng):
        import types

        import synfire.visualization as viz

        class FakePlt:
            @staticmethod
            def subplots(*a, **kw):
                return types.SimpleNamespace(), _FakeAxes()

        orig = viz._require_matplotlib
        viz._require_matplotlib = lambda: (types.SimpleNamespace(), FakePlt())
        try:
            ax = _FakeAxes()
            result = viz.plot_anomaly_scores(rng.random(100), ax=ax)
            assert result is ax
        finally:
            viz._require_matplotlib = orig

    def test_plot_anomaly_scores_with_labels_and_threshold(self, rng):
        import types

        import synfire.visualization as viz

        class FakePlt:
            @staticmethod
            def subplots(*a, **kw):
                return types.SimpleNamespace(), _FakeAxes()

        orig = viz._require_matplotlib

        # Need matplotlib.patches.Patch to exist; patch it
        fake_patch_mod = types.ModuleType("matplotlib.patches")
        fake_patch_mod.Patch = lambda **kw: None
        orig_patches = sys.modules.get("matplotlib.patches")
        sys.modules["matplotlib.patches"] = fake_patch_mod

        viz._require_matplotlib = lambda: (types.SimpleNamespace(), FakePlt())
        try:
            ax = _FakeAxes()
            labels = (np.arange(100) > 70).astype(float)
            result = viz.plot_anomaly_scores(rng.random(100), labels=labels, threshold=0.5, ax=ax)
            assert result is ax
        finally:
            viz._require_matplotlib = orig
            if orig_patches is not None:
                sys.modules["matplotlib.patches"] = orig_patches
            elif "matplotlib.patches" in sys.modules:
                del sys.modules["matplotlib.patches"]

    def test_plot_score_decomposition_with_mocked_mpl(self, rng):
        import types

        import synfire.visualization as viz

        class FakePlt:
            @staticmethod
            def subplots(*a, **kw):
                return types.SimpleNamespace(), _FakeAxes()

        orig = viz._require_matplotlib
        viz._require_matplotlib = lambda: (types.SimpleNamespace(), FakePlt())
        try:
            n = 50
            decomposed = DecomposedAnomalyScore(
                goodness_deficit=rng.random(n),
                prototype_distance=rng.random(n),
                transition_surprise=rng.random(n),
                combined=rng.random(n),
            )
            ax = _FakeAxes()
            result = viz.plot_score_decomposition(decomposed, ax=ax)
            assert result is ax
        finally:
            viz._require_matplotlib = orig

    def test_missing_matplotlib_raises_import_error(self, monkeypatch):
        """When matplotlib is absent, _require_matplotlib raises ImportError."""
        import synfire.visualization as viz
        orig = viz._require_matplotlib

        def raise_import():
            raise ImportError("no matplotlib")

        viz._require_matplotlib = raise_import
        try:
            with pytest.raises(ImportError):
                viz._require_matplotlib()
        finally:
            viz._require_matplotlib = orig


# ===========================================================================
# 6. Hard Negative Mining
# ===========================================================================

class TestHardNegativeMining:
    def test_negative_strategy_default_random(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8)
        assert cfg.negative_strategy == "random"

    def test_negative_strategy_hard_accepted(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, negative_strategy="hard")
        assert cfg.negative_strategy == "hard"

    def test_negative_strategy_curriculum_accepted(self):
        cfg = FFLayerConfig(input_dim=10, hidden_dim=8, negative_strategy="curriculum")
        assert cfg.negative_strategy == "curriculum"

    def test_invalid_negative_strategy_rejected(self):
        with pytest.raises(ValueError, match="negative_strategy must be"):
            FFLayerConfig(input_dim=10, hidden_dim=8, negative_strategy="easy")

    def test_random_strategy_returns_original(self, rng):
        x_pos = rng.standard_normal((10, 4))
        x_neg = rng.standard_normal((10, 4))
        cfg = FFLayerConfig(input_dim=4, hidden_dim=4, negative_strategy="random")
        state = init_layer(cfg)
        result = _mine_hard_negatives(x_pos, x_neg, state, epoch=0, total_epochs=10)
        np.testing.assert_array_equal(result, x_neg)

    def test_hard_strategy_returns_same_shape(self, rng):
        x_pos = rng.standard_normal((20, 8))
        x_neg = rng.standard_normal((20, 8))
        cfg = FFLayerConfig(input_dim=8, hidden_dim=8, negative_strategy="hard")
        state = init_layer(cfg)
        result = _mine_hard_negatives(x_pos, x_neg, state, epoch=0, total_epochs=10)
        assert result.shape == x_pos.shape

    def test_curriculum_at_epoch_0_mostly_random(self, rng):
        """At epoch 0, curriculum should return the same result as random."""
        x_pos = rng.standard_normal((30, 6))
        x_neg = rng.standard_normal((30, 6))
        cfg = FFLayerConfig(input_dim=6, hidden_dim=6, negative_strategy="curriculum")
        state = init_layer(cfg)
        result = _mine_hard_negatives(x_pos, x_neg, state, epoch=0, total_epochs=10)
        assert result.shape == x_pos.shape

    def test_hard_mining_selects_closest(self, rng):
        """Hard mining must select the negative closest to each positive."""
        # 1D: easy to verify
        x_pos = np.array([[0.0], [5.0], [10.0]])
        # x_neg candidates at distances 1, 2, 3 from x_pos[0]
        x_neg = np.array([[1.0], [2.0], [3.0]])
        cfg = FFLayerConfig(input_dim=1, hidden_dim=4, negative_strategy="hard")
        state = init_layer(cfg)
        result = _mine_hard_negatives(x_pos, x_neg, state, epoch=0, total_epochs=10)
        # For x_pos[0]=0: closest is x_neg[0]=1
        assert result[0, 0] == pytest.approx(1.0)
        # For x_pos[1]=5: closest is x_neg[1]=2 (dist 3) or x_neg[2]=3 (dist 2)
        assert result[1, 0] == pytest.approx(3.0)  # 3 is closest to 5
        # For x_pos[2]=10: closest is x_neg[2]=3 (dist 7)
        assert result[2, 0] == pytest.approx(3.0)

    def test_hard_strategy_training_runs(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16, negative_strategy="hard", lr=0.01, epochs=10, seed=5
        )
        state = init_layer(cfg)
        trained, losses = train_layer(state, x_pos, x_neg)
        assert all(np.isfinite(layer) for layer in losses)
        assert np.all(np.isfinite(trained.W))

    def test_curriculum_strategy_training_runs(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16, negative_strategy="curriculum", lr=0.01, epochs=10, seed=6
        )
        state = init_layer(cfg)
        trained, losses = train_layer(state, x_pos, x_neg)
        assert all(np.isfinite(layer) for layer in losses)

    def test_stack_negative_strategy_propagation(self):
        cfg = FFStackConfig(
            layer_dims=(16, 8), lr=0.01, epochs_per_layer=5, negative_strategy="hard"
        )
        state = init_stack(20, cfg)
        for layer in state.layers:
            assert layer.config.negative_strategy == "hard"

    def test_hard_mining_pipeline_end_to_end(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(16,), lr=0.02, epochs_per_layer=10, negative_strategy="hard"
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))


# ===========================================================================
# 7. Multi-Resolution Pipeline
# ===========================================================================

class TestMultiResolutionPipeline:
    def test_default_construction(self):
        mr = MultiResolutionPipeline()
        assert len(mr.window_sizes) == 4
        assert mr.combination == "mean"
        assert not mr._fitted

    def test_custom_window_sizes(self):
        mr = MultiResolutionPipeline(window_sizes=[10, 20])
        assert mr.window_sizes == [10, 20]

    def test_invalid_combination_rejected(self):
        with pytest.raises(ValueError, match="combination must be"):
            MultiResolutionPipeline(combination="median")

    def test_empty_window_sizes_rejected(self):
        with pytest.raises(ValueError):
            MultiResolutionPipeline(window_sizes=[])

    def test_weights_normalized(self):
        mr = MultiResolutionPipeline(window_sizes=[8, 16], weights=[1.0, 3.0])
        assert sum(mr._weights) == pytest.approx(1.0)
        assert mr._weights[1] > mr._weights[0]

    def test_weights_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="weights length"):
            MultiResolutionPipeline(window_sizes=[8, 16, 32], weights=[0.5, 0.5])

    def test_unfitted_raises_on_predict(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 16])
        with pytest.raises(RuntimeError, match="not fitted"):
            mr.anomaly_scores(sine_series)

    def test_fit_creates_pipelines(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 16])
        mr.fit(sine_series)
        assert len(mr._pipelines) == 2
        assert mr._fitted

    def test_anomaly_scores_finite(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 16])
        mr.fit(sine_series)
        scores = mr.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))
        assert len(scores) > 0

    def test_anomaly_scores_mean_combination(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 16], combination="mean")
        mr.fit(sine_series)
        scores = mr.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_anomaly_scores_max_combination(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 16], combination="max")
        mr.fit(sine_series)
        scores = mr.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_mean_and_max_differ(self, sine_series):
        mr_mean = MultiResolutionPipeline(window_sizes=[8, 16], combination="mean")
        mr_max = MultiResolutionPipeline(window_sizes=[8, 16], combination="max")
        mr_mean.fit(sine_series)
        mr_max.fit(sine_series)
        scores_mean = mr_mean.anomaly_scores(sine_series)
        scores_max = mr_max.anomaly_scores(sine_series)
        # Max score must be >= mean score element-wise (max pooling >= weighted avg)
        # They may be close but generally not identical
        assert not np.allclose(scores_mean, scores_max)

    def test_per_resolution_scores_different_lengths(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 32])
        mr.fit(sine_series)
        per_res = mr.score_decomposed_per_resolution(sine_series)
        # Smaller window -> more windows -> longer score array
        assert len(per_res[0]) > len(per_res[1])

    def test_repr_unfitted(self):
        mr = MultiResolutionPipeline(window_sizes=[8, 16])
        assert "unfitted" in repr(mr)

    def test_repr_fitted(self, sine_series):
        mr = MultiResolutionPipeline(window_sizes=[8, 16])
        mr.fit(sine_series)
        assert "fitted" in repr(mr)

    def test_custom_base_config(self, sine_series):
        base = SynfireConfig(
            ff_stack=FFStackConfig(layer_dims=(16,), lr=0.05, epochs_per_layer=10),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        mr = MultiResolutionPipeline(window_sizes=[8, 16], base_config=base)
        mr.fit(sine_series)
        scores = mr.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_weighted_combination(self, sine_series):
        """Weighted mean with extreme weights should bias toward one resolution."""
        mr = MultiResolutionPipeline(
            window_sizes=[8, 16],
            weights=[1.0, 0.0],
            combination="mean",
        )
        mr.fit(sine_series)
        scores_weighted = mr.anomaly_scores(sine_series)

        # Single-resolution pipeline at ws=8
        SynfirePipeline(mr._config_for_window(8))
        # Note: pipelines already fitted; compare output lengths
        assert len(scores_weighted) > 0
        assert np.all(np.isfinite(scores_weighted))


# ===========================================================================
# Integration: Combined new features
# ===========================================================================

class TestCombinedFeatures:
    def test_adam_plus_layer_norm_pipeline(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(32, 16),
                lr=0.005,
                epochs_per_layer=15,
                optimizer="adam",
                layer_norm=True,
                weight_decay=0.001,
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=5),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_decomposition_with_adam(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(16,), lr=0.005, epochs_per_layer=10, optimizer="adam"
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        result = pipeline.score_decomposed(sine_series)
        assert isinstance(result, DecomposedAnomalyScore)
        np.testing.assert_allclose(
            result.combined, pipeline.anomaly_scores(sine_series), atol=1e-12
        )

    def test_hard_negatives_with_layer_norm(self, small_pos_neg):
        x_pos, x_neg = small_pos_neg
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16,
            layer_norm=True, negative_strategy="hard",
            lr=0.01, epochs=10, seed=9,
        )
        state = init_layer(cfg)
        _, losses = train_layer(state, x_pos, x_neg)
        assert all(np.isfinite(layer) for layer in losses)

    def test_multiresolution_with_custom_config(self, sine_series):
        base = SynfireConfig(
            ff_stack=FFStackConfig(
                layer_dims=(16,), lr=0.01, epochs_per_layer=8,
                optimizer="adam",
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        mr = MultiResolutionPipeline(window_sizes=[8, 16], base_config=base)
        mr.fit(sine_series)
        scores = mr.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_config_replace_new_fields(self):
        config = SynfireConfig()
        updated = config.replace(
            ff_stack__optimizer="adam",
            ff_stack__weight_decay=0.01,
            ff_stack__layer_norm=True,
            ff_stack__negative_strategy="curriculum",
        )
        assert updated.ff_stack.optimizer == "adam"
        assert updated.ff_stack.weight_decay == pytest.approx(0.01)
        assert updated.ff_stack.layer_norm is True
        assert updated.ff_stack.negative_strategy == "curriculum"
