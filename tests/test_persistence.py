import numpy as np
import pytest

from synfire import SynfireConfig, SynfirePipeline
from synfire.core.config import FFStackConfig, HebbianConfig, WindowConfig


@pytest.fixture
def small_config():
    return SynfireConfig(
        window=WindowConfig(window_size=20, stride=1),
        ff_stack=FFStackConfig(layer_dims=(32, 16), lr=0.01, epochs_per_layer=30),
        hebbian=HebbianConfig(n_prototypes=4, lr=0.05, inhibition_strength=0.01, epochs=10),
    )


class TestPersistence:
    def test_save_load_roundtrip(self, sine_series, small_config, tmp_path):
        pipeline = SynfirePipeline(small_config)
        pipeline.fit(sine_series)

        scores_before = pipeline.anomaly_scores(sine_series)
        clusters_before = pipeline.cluster(sine_series)

        save_path = tmp_path / "model.npz"
        pipeline.save(save_path)

        loaded = SynfirePipeline.load(save_path)
        scores_after = loaded.anomaly_scores(sine_series)
        clusters_after = loaded.cluster(sine_series)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-12)
        np.testing.assert_array_equal(clusters_before, clusters_after)

    def test_save_unfitted_raises(self, tmp_path):
        pipeline = SynfirePipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.save(tmp_path / "model.npz")

    def test_loaded_pipeline_has_correct_config(self, sine_series, small_config, tmp_path):
        pipeline = SynfirePipeline(small_config)
        pipeline.fit(sine_series)

        save_path = tmp_path / "model.npz"
        pipeline.save(save_path)

        loaded = SynfirePipeline.load(save_path)
        assert loaded.config.window.window_size == small_config.window.window_size
        assert loaded.config.ff_stack.layer_dims == small_config.ff_stack.layer_dims
        assert loaded.config.hebbian.n_prototypes == small_config.hebbian.n_prototypes
        assert loaded._fitted is True

    def test_save_invalid_type_raises(self, tmp_path):
        from synfire.persistence import save_pipeline

        with pytest.raises(TypeError, match="Expected SynfirePipeline"):
            save_pipeline("not_a_pipeline", tmp_path / "model.npz")

    def test_save_corrupted_state_raises(self, tmp_path):
        from synfire.persistence import save_pipeline

        pipeline = SynfirePipeline()
        pipeline._fitted = True
        with pytest.raises(RuntimeError, match="Pipeline state is corrupted"):
            save_pipeline(pipeline, tmp_path / "model.npz")

