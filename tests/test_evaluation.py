import numpy as np
import pytest

from synfire import SynfireConfig
from synfire.core.config import FFStackConfig, HebbianConfig, WindowConfig
from synfire.evaluation import EvaluationResult, evaluate_multi_seed, mann_whitney_auc


class TestMannWhitneyAUC:
    def test_perfect_separation(self):
        scores = np.array([10.0, 9.0, 8.0, 1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        auc = mann_whitney_auc(scores, labels)
        assert auc == 1.0

    def test_inverse_separation(self):
        scores = np.array([1.0, 2.0, 3.0, 10.0, 9.0, 8.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        auc = mann_whitney_auc(scores, labels)
        assert auc == 0.0

    def test_random_auc_near_half(self):
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(1000)
        labels = np.zeros(1000)
        labels[:500] = 1
        auc = mann_whitney_auc(scores, labels)
        assert 0.4 < auc < 0.6

    def test_no_positives_returns_half(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 0])
        assert mann_whitney_auc(scores, labels) == 0.5

    def test_no_negatives_returns_half(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1])
        assert mann_whitney_auc(scores, labels) == 0.5


class TestLabelScoreAlignment:
    def test_label_shorter_than_scores_aligns_correctly(self):
        scores = np.array([10.0, 9.0, 8.0, 1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1, 0])  # shorter than scores
        auc = mann_whitney_auc(scores[:4], labels)
        assert 0.0 <= auc <= 1.0

    def test_scores_shorter_than_labels_aligns_correctly(self):
        scores = np.array([10.0, 9.0, 1.0])
        labels = np.array([1, 1, 0, 0, 0])  # longer than scores
        auc = mann_whitney_auc(scores, labels[:3])
        assert 0.0 <= auc <= 1.0


class TestMultiSeedEvaluation:
    @pytest.fixture
    def small_config(self):
        return SynfireConfig(
            window=WindowConfig(window_size=20, stride=1),
            ff_stack=FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=20),
            hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=5),
        )

    def test_multi_seed_different_results(self, sine_series, small_config):
        n_windows = (len(sine_series) - 20) // 1 + 1 - 1
        labels = np.zeros(n_windows)
        # Mark some positions as anomalous
        labels[100:120] = 1
        labels[300:320] = 1

        result = evaluate_multi_seed(
            sine_series, sine_series, labels,
            config=small_config, n_seeds=3,
        )

        assert isinstance(result, EvaluationResult)
        assert len(result.seeds) == 3
        assert len(result.auc_scores) == 3
        assert all(0.0 <= auc <= 1.0 for auc in result.auc_scores)
        assert 0.0 <= result.mean_auc <= 1.0
        assert result.std_auc >= 0.0

    def test_multi_seed_varies_hebbian_seed(self, sine_series, small_config):
        """Each seed should produce a different Hebbian configuration."""
        from synfire.evaluation import _config_with_seed

        c1 = _config_with_seed(small_config, 0)
        c2 = _config_with_seed(small_config, 1)
        assert c1.hebbian.seed != c2.hebbian.seed
        assert c1.ff_stack.seed != c2.ff_stack.seed

    def test_result_post_init(self):
        result = EvaluationResult(seeds=[0, 1, 2], auc_scores=[0.7, 0.8, 0.9])
        assert result.mean_auc == pytest.approx(0.8)
        assert result.std_auc > 0
