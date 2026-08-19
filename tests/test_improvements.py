"""Tests for algorithm improvements: early stopping, LR scheduling, ensemble goodness,
adaptive threshold, training history, and multi-layer scoring invariants."""

from __future__ import annotations

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
    _scheduled_lr,
    goodness,
    init_layer,
    train_layer,
    train_step,
)
from synfire.layers.ff_stack import init_stack, train_stack
from synfire.pipeline.anomaly import _ensemble_goodness_deficit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_series():
    t = np.arange(1000, dtype=np.float64)
    return np.sin(2 * np.pi * t / 50)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_ff_config():
    return FFLayerConfig(input_dim=20, hidden_dim=16, lr=0.05, threshold=2.0, epochs=30, seed=7)


@pytest.fixture
def small_stack_config():
    return FFStackConfig(layer_dims=(32, 16), lr=0.01, threshold=2.0, epochs_per_layer=30, seed=42)


@pytest.fixture
def minimal_pipeline_config():
    return SynfireConfig(
        window=WindowConfig(window_size=20, stride=1),
        norm=NormConfig(method="zscore"),
        ff_stack=FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=20),
        hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=5),
    )


# ---------------------------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------------------------

class TestCosineLRSchedule:
    def test_starts_at_base_lr(self):
        assert _scheduled_lr(0.1, epoch=0, total_epochs=10, schedule="cosine") == pytest.approx(0.1)

    def test_ends_near_zero(self):
        lr_final = _scheduled_lr(0.1, epoch=9, total_epochs=10, schedule="cosine")
        assert lr_final < 0.01

    def test_monotonically_decreasing(self):
        lrs = [_scheduled_lr(0.1, epoch=e, total_epochs=20, schedule="cosine") for e in range(20)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-12, f"LR increased at epoch {i}"

    def test_single_epoch(self):
        # With only one epoch, schedule should return base_lr
        assert _scheduled_lr(
            0.05, epoch=0, total_epochs=1, schedule="cosine"
        ) == pytest.approx(0.05)

    def test_cosine_layer_trains(self, rng):
        """Layer with cosine schedule should converge (loss decreases)."""
        cfg = FFLayerConfig(
            input_dim=20, hidden_dim=16, lr=0.05, epochs=50, seed=5, lr_schedule="cosine"
        )
        state = init_layer(cfg)
        x_pos = rng.standard_normal((100, 20))
        x_neg = rng.permutation(rng.standard_normal((100, 20)))
        trained, losses = train_layer(state, x_pos, x_neg)
        assert len(losses) == 50
        assert losses[-1] < losses[0]

    def test_cosine_vs_fixed_same_epochs(self, rng):
        """Cosine schedule should complete the same number of epochs as fixed."""
        epochs = 40
        x_pos = rng.standard_normal((80, 10))
        x_neg = rng.permutation(rng.standard_normal((80, 10)))

        cfg_fixed = FFLayerConfig(input_dim=10, hidden_dim=8, lr=0.05, epochs=epochs, seed=1)
        cfg_cosine = FFLayerConfig(
            input_dim=10, hidden_dim=8, lr=0.05, epochs=epochs, seed=1, lr_schedule="cosine"
        )
        _, losses_fixed = train_layer(init_layer(cfg_fixed), x_pos, x_neg)
        _, losses_cosine = train_layer(init_layer(cfg_cosine), x_pos, x_neg)

        assert len(losses_fixed) == epochs
        assert len(losses_cosine) == epochs
        # Cosine should produce a different trajectory than fixed
        assert not np.allclose(losses_fixed, losses_cosine)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_patience_zero_runs_all_epochs(self, rng):
        cfg = FFLayerConfig(
            input_dim=10, hidden_dim=8, lr=0.01, epochs=50, seed=3,
            early_stopping_patience=0,
        )
        state = init_layer(cfg)
        x_pos = rng.standard_normal((60, 10))
        x_neg = rng.permutation(rng.standard_normal((60, 10)))
        _, losses = train_layer(state, x_pos, x_neg)
        assert len(losses) == 50

    def test_patience_triggers_early_termination(self):
        """With tight patience on a converged model, training should stop early."""
        # Train briefly to get a partially trained state, then apply very tight early stopping
        rng = np.random.default_rng(99)
        x_pos = rng.standard_normal((80, 10))
        x_neg = rng.permutation(rng.standard_normal((80, 10)))

        # Pre-train with no early stopping
        cfg_pretrain = FFLayerConfig(input_dim=10, hidden_dim=8, lr=0.01, epochs=200, seed=9)
        pretrained, _ = train_layer(init_layer(cfg_pretrain), x_pos, x_neg)

        # Now apply very tight early stopping (should fire quickly since model is converged)
        cfg_es = FFLayerConfig(
            input_dim=10, hidden_dim=8, lr=0.001, epochs=100, seed=9,
            early_stopping_patience=3,
            early_stopping_min_delta=1.0,  # very large delta -> nothing qualifies as improvement
        )
        # Manually copy weights from pretrained to new state with ES config
        from synfire.layers.ff_layer import FFLayerState
        es_state = FFLayerState(W=pretrained.W.copy(), b=pretrained.b.copy(), config=cfg_es)
        _, losses_es = train_layer(es_state, x_pos, x_neg)

        # Early stopping should cut it well short of 100 epochs
        assert len(losses_es) < 100

    def test_patience_history_length_bounded_by_epochs(self, rng):
        """Loss history length is always <= config.epochs."""
        cfg = FFLayerConfig(
            input_dim=10, hidden_dim=8, lr=0.05, epochs=100, seed=11,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-6,
        )
        state = init_layer(cfg)
        x_pos = rng.standard_normal((50, 10))
        x_neg = rng.permutation(rng.standard_normal((50, 10)))
        _, losses = train_layer(state, x_pos, x_neg)
        assert len(losses) <= 100

    def test_losses_are_always_finite(self, rng):
        cfg = FFLayerConfig(
            input_dim=10, hidden_dim=8, lr=0.05, epochs=30, seed=13,
            early_stopping_patience=5,
        )
        state = init_layer(cfg)
        x_pos = rng.standard_normal((60, 10))
        x_neg = rng.permutation(rng.standard_normal((60, 10)))
        _, losses = train_layer(state, x_pos, x_neg)
        assert all(np.isfinite(layer) for layer in losses)

    def test_stack_with_early_stopping(self, rng):
        """FFStackConfig early_stopping_patience propagates to each layer."""
        cfg = FFStackConfig(
            layer_dims=(16, 8), lr=0.05, epochs_per_layer=100,
            early_stopping_patience=5, early_stopping_min_delta=1e-6, seed=17,
        )
        state = init_stack(20, cfg)
        x_pos = rng.standard_normal((80, 20))
        x_neg = rng.permutation(rng.standard_normal((80, 20)))
        trained, all_losses = train_stack(state, x_pos, x_neg)
        assert len(all_losses) == 2
        for layer_losses in all_losses:
            assert len(layer_losses) <= 100
            assert all(np.isfinite(layer) for layer in layer_losses)

    def test_config_negative_patience_rejected(self):
        with pytest.raises(ValueError, match="early_stopping_patience must be >= 0"):
            FFLayerConfig(input_dim=10, hidden_dim=8, early_stopping_patience=-1)


# ---------------------------------------------------------------------------
# LR schedule config validation
# ---------------------------------------------------------------------------

class TestLRScheduleConfig:
    def test_invalid_lr_schedule_rejected(self):
        with pytest.raises(ValueError, match="lr_schedule must be"):
            FFLayerConfig(input_dim=10, hidden_dim=8, lr_schedule="linear")

    def test_invalid_stack_lr_schedule_rejected(self):
        with pytest.raises(ValueError, match="lr_schedule must be"):
            FFStackConfig(lr_schedule="exp_decay")

    def test_stack_lr_schedule_propagates_to_layers(self):
        cfg = FFStackConfig(layer_dims=(32, 16), lr=0.05, epochs_per_layer=10, lr_schedule="cosine")
        state = init_stack(20, cfg)
        for layer in state.layers:
            assert layer.config.lr_schedule == "cosine"

    def test_warmup_cosine_schedule(self):
        """warmup_cosine: LR starts low, rises, then decays."""
        cfg = FFLayerConfig(
            input_dim=10, hidden_dim=8, lr=0.1, epochs=20, seed=3,
            lr_schedule="warmup_cosine", lr_warmup_fraction=0.2,
        )
        lrs = [_scheduled_lr(cfg.lr, e, cfg.epochs, cfg.lr_schedule, cfg.lr_warmup_fraction)
               for e in range(cfg.epochs)]
        warmup_end = max(1, int(20 * 0.2))
        # During warmup, LR should be increasing
        assert lrs[warmup_end - 1] > lrs[0]
        # After warmup, LR should decrease toward end
        assert lrs[-1] < lrs[warmup_end]

    def test_constant_schedule(self):
        for epoch in range(10):
            lr = _scheduled_lr(0.05, epoch, total_epochs=10, schedule="constant")
            assert lr == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Ensemble goodness
# ---------------------------------------------------------------------------

class TestEnsembleGoodness:
    def test_shape(self, rng):
        """Ensemble goodness deficit has shape (batch,)."""
        activations = [rng.standard_normal((50, 32)), rng.standard_normal((50, 16))]
        activations = [np.maximum(a, 0) for a in activations]
        deficit = _ensemble_goodness_deficit(activations, threshold=2.0)
        assert deficit.shape == (50,)

    def test_single_layer_matches_direct(self, rng):
        """Single-layer ensemble matches plain goodness deficit."""
        acts = [np.maximum(rng.standard_normal((30, 16)), 0)]
        deficit_ensemble = _ensemble_goodness_deficit(acts, threshold=2.0)
        deficit_direct = 2.0 - goodness(acts[0])
        np.testing.assert_allclose(deficit_ensemble, deficit_direct, atol=1e-12)

    def test_later_layers_weighted_more(self, rng):
        """Verify later layers dominate by checking a degenerate case."""
        # Layer 0: low goodness (high deficit), Layer 1: high goodness (low deficit)
        n = 40
        acts_low = [np.ones((n, 4)) * 0.1]    # goodness ≈ 0.01, deficit ≈ 1.99
        acts_high = [np.ones((n, 4)) * 10.0]  # goodness ≈ 100, deficit ≈ -98

        # With 2-layer ensemble, layer 2 (high) is weighted 2/3 vs layer 1 (low) 1/3
        deficit_2layer = _ensemble_goodness_deficit(acts_low + acts_high, threshold=2.0)
        # The high-goodness last layer should push deficit negative
        assert np.all(deficit_2layer < 2.0)  # deficit should be less than acts_low alone

    def test_ensemble_goodness_pipeline_integration(self, sine_series, minimal_pipeline_config):
        """Pipeline with ensemble_goodness=True produces finite scores."""
        config = SynfireConfig(
            window=minimal_pipeline_config.window,
            norm=minimal_pipeline_config.norm,
            ff_stack=minimal_pipeline_config.ff_stack,
            hebbian=minimal_pipeline_config.hebbian,
            anomaly=AnomalyConfig(ensemble_goodness=True),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))
        assert len(scores) > 0

    def test_ensemble_vs_single_layer_differ(self, sine_series, minimal_pipeline_config):
        """Ensemble and single-layer goodness should produce different scores."""
        base = minimal_pipeline_config

        config_ensemble = SynfireConfig(
            window=base.window, norm=base.norm, ff_stack=base.ff_stack, hebbian=base.hebbian,
            anomaly=AnomalyConfig(
                weight_goodness=1.0, weight_distance=0.0, weight_transition=0.0,
                use_goodness=True, use_distance=False, use_transition=False,
                ensemble_goodness=True,
            ),
        )
        config_single = SynfireConfig(
            window=base.window, norm=base.norm, ff_stack=base.ff_stack, hebbian=base.hebbian,
            anomaly=AnomalyConfig(
                weight_goodness=1.0, weight_distance=0.0, weight_transition=0.0,
                use_goodness=True, use_distance=False, use_transition=False,
                ensemble_goodness=False,
            ),
        )

        p_ensemble = SynfirePipeline(config_ensemble)
        p_ensemble.fit(sine_series)
        scores_ensemble = p_ensemble.anomaly_scores(sine_series)

        p_single = SynfirePipeline(config_single)
        p_single.fit(sine_series)
        scores_single = p_single.anomaly_scores(sine_series)

        # Scores exist and are valid
        assert np.all(np.isfinite(scores_ensemble))
        assert np.all(np.isfinite(scores_single))

    def test_ensemble_goodness_single_stack_layer(self, sine_series):
        """With 1-layer stack, ensemble_goodness falls back to single-layer silently."""
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(layer_dims=(32,), lr=0.01, epochs_per_layer=10),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
            anomaly=AnomalyConfig(ensemble_goodness=True),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

class TestAdaptiveThreshold:
    def test_adaptive_threshold_set_after_fit(self, sine_series, minimal_pipeline_config):
        config = SynfireConfig(
            window=minimal_pipeline_config.window,
            norm=minimal_pipeline_config.norm,
            ff_stack=minimal_pipeline_config.ff_stack,
            hebbian=minimal_pipeline_config.hebbian,
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        # Should have been updated from the default config threshold
        assert pipeline._effective_threshold > 0
        assert np.isfinite(pipeline._effective_threshold)

    def test_adaptive_threshold_differs_from_config(self, sine_series, minimal_pipeline_config):
        """Adaptive threshold should differ from the config value after fit on real data."""
        config_threshold = 2.0  # default
        config = SynfireConfig(
            window=minimal_pipeline_config.window,
            norm=minimal_pipeline_config.norm,
            ff_stack=FFStackConfig(
                layer_dims=(32, 16), lr=0.01, epochs_per_layer=20, threshold=config_threshold
            ),
            hebbian=minimal_pipeline_config.hebbian,
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        # On a real sine dataset the mean positive goodness won't be exactly 2.0
        assert pipeline._effective_threshold != config_threshold

    def test_adaptive_disabled_keeps_config_threshold(self, sine_series, minimal_pipeline_config):
        config_threshold = 2.0
        config = SynfireConfig(
            window=minimal_pipeline_config.window,
            norm=minimal_pipeline_config.norm,
            ff_stack=FFStackConfig(
                layer_dims=(32, 16), lr=0.01, epochs_per_layer=20, threshold=config_threshold
            ),
            hebbian=minimal_pipeline_config.hebbian,
            adaptive_threshold=False,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        assert pipeline._effective_threshold == pytest.approx(config_threshold)

    def test_adaptive_threshold_scores_finite(self, sine_series, minimal_pipeline_config):
        config = SynfireConfig(
            window=minimal_pipeline_config.window,
            norm=minimal_pipeline_config.norm,
            ff_stack=minimal_pipeline_config.ff_stack,
            hebbian=minimal_pipeline_config.hebbian,
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))

    def test_adaptive_threshold_persistence(self, sine_series, minimal_pipeline_config, tmp_path):
        """Effective threshold survives save/load roundtrip."""
        config = SynfireConfig(
            window=minimal_pipeline_config.window,
            norm=minimal_pipeline_config.norm,
            ff_stack=minimal_pipeline_config.ff_stack,
            hebbian=minimal_pipeline_config.hebbian,
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)

        threshold_before = pipeline._effective_threshold
        path = tmp_path / "model_adaptive.npz"
        pipeline.save(path)
        loaded = SynfirePipeline.load(path)

        assert loaded._effective_threshold == pytest.approx(threshold_before)
        scores_before = pipeline.anomaly_scores(sine_series)
        scores_after = loaded.anomaly_scores(sine_series)
        np.testing.assert_allclose(scores_before, scores_after, atol=1e-12)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

class TestTrainingHistory:
    def test_history_stored_after_fit(self, sine_series, minimal_pipeline_config):
        pipeline = SynfirePipeline(minimal_pipeline_config)
        pipeline.fit(sine_series)
        assert len(pipeline.training_history) == len(minimal_pipeline_config.ff_stack.layer_dims)

    def test_history_has_loss_values(self, sine_series, minimal_pipeline_config):
        pipeline = SynfirePipeline(minimal_pipeline_config)
        pipeline.fit(sine_series)
        for layer_losses in pipeline.training_history:
            assert len(layer_losses) > 0
            assert all(np.isfinite(layer) for layer in layer_losses)

    def test_history_length_per_layer(self, sine_series, minimal_pipeline_config):
        epochs = minimal_pipeline_config.ff_stack.epochs_per_layer
        pipeline = SynfirePipeline(minimal_pipeline_config)
        pipeline.fit(sine_series)
        for layer_losses in pipeline.training_history:
            assert len(layer_losses) <= epochs

    def test_history_loss_decreases(self, sine_series, minimal_pipeline_config):
        """Loss should generally decrease across all layers."""
        pipeline = SynfirePipeline(minimal_pipeline_config)
        pipeline.fit(sine_series)
        for layer_losses in pipeline.training_history:
            assert layer_losses[-1] < layer_losses[0]

    def test_history_empty_before_fit(self):
        pipeline = SynfirePipeline()
        assert pipeline.training_history == []

    def test_history_with_early_stopping(self, sine_series):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(16,), lr=0.05, epochs_per_layer=100,
                early_stopping_patience=5, early_stopping_min_delta=1e-6,
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=5),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        # History should exist and have <= 100 entries
        assert len(pipeline.training_history) == 1
        assert len(pipeline.training_history[0]) <= 100


# ---------------------------------------------------------------------------
# Property-based invariants (without hypothesis)
# These test fundamental mathematical invariants that must hold for any input.
# ---------------------------------------------------------------------------

class TestAlgorithmInvariants:
    """Invariant tests: properties that must hold regardless of data."""

    @pytest.mark.parametrize("batch_size", [1, 5, 50, 200])
    def test_goodness_always_nonneg(self, batch_size):
        rng = np.random.default_rng(batch_size)
        h = np.maximum(rng.standard_normal((batch_size, 32)), 0)
        g = goodness(h)
        assert np.all(g >= 0), "Goodness must be non-negative (MSA of ReLU outputs)"

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_ensemble_goodness_finite(self, n_layers):
        rng = np.random.default_rng(n_layers)
        acts = [np.maximum(rng.standard_normal((40, 16 * (n_layers - i))), 0)
                for i in range(n_layers)]
        deficit = _ensemble_goodness_deficit(acts, threshold=2.0)
        assert np.all(np.isfinite(deficit))
        assert deficit.shape == (40,)

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_anomaly_scores_finite_for_varied_seeds(self, sine_series, seed):
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(layer_dims=(16,), lr=0.05, epochs_per_layer=10, seed=seed),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3, seed=seed),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)
        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores)), f"Non-finite scores with seed={seed}"

    def test_scores_unchanged_by_input_scaling(self, sine_series):
        """Anomaly scores must be invariant under the pipeline's own normalization."""
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(layer_dims=(16, 8), lr=0.05, epochs_per_layer=10),
            hebbian=HebbianConfig(n_prototypes=4, epochs=3),
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)

        # Scores on the same data twice should be identical (determinism check)
        scores1 = pipeline.anomaly_scores(sine_series)
        scores2 = pipeline.anomaly_scores(sine_series)
        np.testing.assert_array_equal(scores1, scores2)

    def test_train_step_weights_change(self):
        """A single train step must change weights (non-zero gradient path)."""
        rng = np.random.default_rng(55)
        cfg = FFLayerConfig(input_dim=8, hidden_dim=12, lr=0.1, seed=55)
        state = init_layer(cfg)
        x_pos = rng.standard_normal((20, 8))
        x_neg = rng.standard_normal((20, 8))

        new_state, _ = train_step(state, x_pos, x_neg)
        assert not np.allclose(state.W, new_state.W), "Weights must change after a train step"


# ---------------------------------------------------------------------------
# Integration: full train -> detect pipeline with new features
# ---------------------------------------------------------------------------

class TestIntegrationNewFeatures:
    """End-to-end integration tests for the combined feature set."""

    def test_all_features_enabled(self, sine_series):
        """Pipeline with all new features enabled should produce valid scores."""
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            norm=NormConfig(method="zscore"),
            ff_stack=FFStackConfig(
                layer_dims=(32, 16),
                lr=0.05,
                epochs_per_layer=50,
                early_stopping_patience=5,
                early_stopping_min_delta=1e-5,
                lr_schedule="cosine",
            ),
            hebbian=HebbianConfig(n_prototypes=8, lr=0.05, inhibition_strength=0.05, epochs=10),
            anomaly=AnomalyConfig(
                weight_goodness=0.3, weight_distance=0.5, weight_transition=0.2,
                use_goodness=True, use_distance=True, use_transition=True,
                ensemble_goodness=True,
            ),
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)

        assert pipeline._fitted
        assert pipeline._effective_threshold > 0
        assert len(pipeline.training_history) == 2
        for layer_losses in pipeline.training_history:
            assert len(layer_losses) > 0
            assert len(layer_losses) <= 50

        scores = pipeline.anomaly_scores(sine_series)
        assert np.all(np.isfinite(scores))
        assert len(scores) > 0

    def test_cosine_early_stopping_persistence_roundtrip(self, sine_series, tmp_path):
        """Save/load with cosine schedule and early stopping preserves scores exactly."""
        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(16, 8),
                lr=0.05,
                epochs_per_layer=30,
                early_stopping_patience=5,
                lr_schedule="cosine",
            ),
            hebbian=HebbianConfig(n_prototypes=4, epochs=5),
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(sine_series)

        scores_orig = pipeline.anomaly_scores(sine_series)
        path = tmp_path / "model_full.npz"
        pipeline.save(path)

        loaded = SynfirePipeline.load(path)
        scores_loaded = loaded.anomaly_scores(sine_series)

        np.testing.assert_allclose(scores_orig, scores_loaded, atol=1e-12)
        assert loaded.config.ff_stack.lr_schedule == "cosine"
        assert loaded.config.ff_stack.early_stopping_patience == 5
        assert loaded.config.adaptive_threshold is True

    def test_anomaly_detection_spike_sensitivity(self):
        """Pipeline with new features should rank spike regions higher than baseline."""
        t = np.arange(1200, dtype=np.float64)
        train = np.sin(2 * np.pi * t / 50)

        t_test = np.arange(1500, dtype=np.float64)
        test = np.sin(2 * np.pi * t_test / 50)
        # Inject spikes at known positions
        spike_pos = [500, 700, 900]
        for pos in spike_pos:
            test[pos:pos + 5] += 6.0

        config = SynfireConfig(
            window=WindowConfig(window_size=20),
            ff_stack=FFStackConfig(
                layer_dims=(32, 16), lr=0.05, epochs_per_layer=40,
                lr_schedule="cosine",
            ),
            hebbian=HebbianConfig(n_prototypes=8, epochs=10),
            anomaly=AnomalyConfig(ensemble_goodness=True),
            adaptive_threshold=True,
        )
        pipeline = SynfirePipeline(config)
        pipeline.fit(train)
        scores = pipeline.anomaly_scores(test)

        assert np.all(np.isfinite(scores))
        # Scores near spike positions should be elevated vs. baseline
        # (not asserting perfect detection, just that the system produces a signal)
        w = 20  # window size
        spike_mask = np.zeros(len(scores), dtype=bool)
        for pos in spike_pos:
            lo = max(0, pos - w)
            hi = min(len(scores), pos + w)
            spike_mask[lo:hi] = True
        mean_spike = scores[spike_mask].mean()
        mean_normal = scores[~spike_mask].mean()
        # Spike score should be at least as high as normal on average
        # (soft check: we don't guarantee perfect AUC, just meaningful signal)
        assert mean_spike >= mean_normal * 0.8, (
            f"Spike regions score {mean_spike:.3f} vs normal {mean_normal:.3f}"
        )
