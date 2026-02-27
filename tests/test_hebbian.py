import numpy as np

from synfire.core.config import HebbianConfig
from synfire.layers.hebbian import (
    assign,
    distances_to_prototypes,
    init_hebbian,
    train_hebbian,
)


def _make_clustered_data(rng, n_per_cluster=100):
    """Create 3 well-separated clusters in 2D."""
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    data = []
    for c in centers:
        data.append(rng.standard_normal((n_per_cluster, 2)) * 0.3 + c)
    return np.vstack(data), centers


class TestHebbianInit:
    def test_prototype_count(self, rng):
        data = rng.standard_normal((100, 16))
        cfg = HebbianConfig(n_prototypes=5)
        state = init_hebbian(data, cfg)
        assert state.prototypes.shape == (5, 16)

    def test_prototypes_from_data(self, rng):
        data = rng.standard_normal((50, 8))
        cfg = HebbianConfig(n_prototypes=3)
        state = init_hebbian(data, cfg)
        # Each prototype should be a point from the data
        for p in state.prototypes:
            dists = np.sum((data - p) ** 2, axis=1)
            assert np.min(dists) < 1e-10


class TestAssign:
    def test_assignment_shape(self, rng):
        data = rng.standard_normal((50, 8))
        cfg = HebbianConfig(n_prototypes=4)
        state = init_hebbian(data, cfg)
        labels = assign(state, data)
        assert labels.shape == (50,)
        assert np.all(labels >= 0) and np.all(labels < 4)

    def test_nearest_prototype(self):
        cfg = HebbianConfig(n_prototypes=2)
        from synfire.layers.hebbian import HebbianState

        state = HebbianState(
            prototypes=np.array([[0.0, 0.0], [10.0, 10.0]]),
            config=cfg,
        )
        x = np.array([[0.1, 0.1], [9.9, 9.9], [0.0, 0.0]])
        labels = assign(state, x)
        np.testing.assert_array_equal(labels, [0, 1, 0])


class TestDistances:
    def test_distance_shape(self, rng):
        data = rng.standard_normal((30, 8))
        cfg = HebbianConfig(n_prototypes=3)
        state = init_hebbian(data, cfg)
        dists = distances_to_prototypes(state, data)
        assert dists.shape == (30,)
        assert np.all(dists >= 0)


class TestTrainHebbian:
    def test_prototypes_converge_to_clusters(self, rng):
        data, true_centers = _make_clustered_data(rng)
        cfg = HebbianConfig(
            n_prototypes=3, lr=0.05, inhibition_strength=0.01, epochs=20, seed=42
        )
        state = init_hebbian(data, cfg)
        state = train_hebbian(state, data, batch_size=32)

        # Each true center should have a nearby prototype
        for center in true_centers:
            dists = np.sqrt(np.sum((state.prototypes - center) ** 2, axis=1))
            assert np.min(dists) < 2.0, f"No prototype near center {center}"

    def test_cluster_assignment_accuracy(self, rng):
        data, true_centers = _make_clustered_data(rng, n_per_cluster=200)
        cfg = HebbianConfig(
            n_prototypes=3, lr=0.05, inhibition_strength=0.01, epochs=30, seed=42
        )
        state = init_hebbian(data, cfg)
        state = train_hebbian(state, data, batch_size=32)

        labels = assign(state, data)
        # Points from the same true cluster should mostly get the same label
        for i in range(3):
            cluster_labels = labels[i * 200 : (i + 1) * 200]
            most_common = np.bincount(cluster_labels).argmax()
            purity = np.mean(cluster_labels == most_common)
            assert purity > 0.8, f"Cluster {i} purity too low: {purity}"
