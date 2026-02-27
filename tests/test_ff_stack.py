import numpy as np

from synfire.core.config import FFStackConfig, WindowConfig
from synfire.layers.ff_stack import (
    extract_representation,
    forward_stack,
    init_stack,
    train_stack,
)
from synfire.preprocessing.windows import (
    make_consecutive_pairs,
    make_random_pairs,
    sliding_windows,
)


class TestFFStackInit:
    def test_layer_count(self):
        cfg = FFStackConfig(layer_dims=(64, 32, 16))
        state = init_stack(40, cfg)
        assert len(state.layers) == 3

    def test_dimension_chaining(self):
        cfg = FFStackConfig(layer_dims=(64, 32))
        state = init_stack(40, cfg)
        assert state.layers[0].W.shape == (64, 40)
        assert state.layers[1].W.shape == (32, 64)


class TestTrainStack:
    def test_loss_per_layer(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)
        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=30)
        state = init_stack(40, cfg)
        trained, all_losses = train_stack(state, x_pos, x_neg)

        assert len(all_losses) == 2
        for losses in all_losses:
            assert len(losses) == 30
            assert losses[-1] < losses[0]


class TestForwardStack:
    def test_activations_per_layer(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)
        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=30)
        state = init_stack(40, cfg)
        trained, _ = train_stack(state, x_pos, x_neg)

        activations = forward_stack(trained, x_pos)
        assert len(activations) == 2
        assert activations[0].shape == (len(x_pos), 32)
        assert activations[1].shape == (len(x_pos), 16)


class TestExtractRepresentation:
    def test_output_shape(self, sine_series, rng):
        windows = sliding_windows(sine_series, WindowConfig(window_size=20))
        pos_l, pos_r = make_consecutive_pairs(windows)
        neg_l, neg_r = make_random_pairs(windows, rng, min_gap=5)
        x_pos = np.concatenate([pos_l, pos_r], axis=1)
        x_neg = np.concatenate([neg_l[: len(pos_l)], neg_r[: len(pos_l)]], axis=1)

        cfg = FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=30)
        state = init_stack(40, cfg)
        trained, _ = train_stack(state, x_pos, x_neg)

        rep = extract_representation(trained, x_pos)
        assert rep.shape == (len(x_pos), 16)
        assert np.all(rep >= 0)  # After ReLU
