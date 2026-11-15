from types import SimpleNamespace

import numpy as np
import pytest

from synfire import visualization


class _Axes:
    def hist(self, *args, **kwargs):
        return None

    def axvline(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return []

    def axhline(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def legend(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("threshold", ["0.5", np.nan, np.inf, True])
def test_plot_anomaly_scores_rejects_invalid_threshold(monkeypatch, threshold):
    monkeypatch.setattr(visualization, "_require_matplotlib", lambda: (None, None))

    with pytest.raises(ValueError, match="threshold must be a finite real number"):
        visualization.plot_anomaly_scores(np.array([0.1, 0.9]), threshold=threshold, ax=_Axes())


@pytest.mark.parametrize("threshold", ["0.5", np.nan, np.inf, True])
def test_plot_goodness_distribution_rejects_invalid_threshold(monkeypatch, threshold):
    monkeypatch.setattr(visualization, "_require_matplotlib", lambda: (None, None))

    with pytest.raises(ValueError, match="threshold must be a finite real number"):
        visualization.plot_goodness_distribution(
            np.array([0.1, 0.2]),
            np.array([0.8, 0.9]),
            threshold=threshold,
            ax=_Axes(),
        )


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        ("pos_goodness", np.ones((1, 2)), "must be 1D"),
        ("neg_goodness", np.ones((1, 2)), "must be 1D"),
        ("pos_goodness", np.array(["bad", "input"]), "must contain real numeric values"),
        ("neg_goodness", np.array(["bad", "input"]), "must contain real numeric values"),
        ("pos_goodness", np.array([1.0 + 1.0j, 2.0]), "must contain real numeric values"),
        ("neg_goodness", np.array([1.0 + 1.0j, 2.0]), "must contain real numeric values"),
        ("pos_goodness", np.array([np.nan, 1.0]), "must contain only finite values"),
        ("neg_goodness", np.array([np.inf, 1.0]), "must contain only finite values"),
    ],
)
def test_plot_goodness_distribution_rejects_invalid_goodness_inputs(
    monkeypatch, name, values, message
):
    monkeypatch.setattr(visualization, "_require_matplotlib", lambda: (None, None))

    goodness = {
        "pos_goodness": np.array([0.1, 0.2]),
        "neg_goodness": np.array([0.8, 0.9]),
    }
    goodness[name] = values

    with pytest.raises(ValueError, match=f"{name}.*{message}"):
        visualization.plot_goodness_distribution(**goodness, ax=_Axes())


def test_plot_score_decomposition_rejects_non_finite_combined_scores(monkeypatch):
    monkeypatch.setattr(visualization, "_require_matplotlib", lambda: (None, None))

    decomposed = SimpleNamespace(
        combined=np.array([0.1, np.nan]),
        goodness_deficit=None,
        prototype_distance=None,
        transition_surprise=None,
    )

    with pytest.raises(ValueError, match="combined must contain only finite values"):
        visualization.plot_score_decomposition(decomposed, ax=_Axes())


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("goodness_deficit", "Goodness Deficit"),
        ("prototype_distance", "Prototype Distance"),
        ("transition_surprise", "Transition Surprise"),
    ],
)
@pytest.mark.parametrize("value", [np.array([1.0, np.nan]), np.array([1.0, np.inf])])
def test_plot_score_decomposition_rejects_non_finite_components(
    monkeypatch, field, label, value
):
    monkeypatch.setattr(visualization, "_require_matplotlib", lambda: (None, None))

    decomposed = SimpleNamespace(
        combined=np.array([0.1, 0.2]),
        goodness_deficit=None,
        prototype_distance=None,
        transition_surprise=None,
    )
    setattr(decomposed, field, value)

    with pytest.raises(ValueError, match=f"{label} must contain only finite values"):
        visualization.plot_score_decomposition(decomposed, ax=_Axes())


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("goodness_deficit", "Goodness Deficit"),
        ("prototype_distance", "Prototype Distance"),
        ("transition_surprise", "Transition Surprise"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [np.array(["bad", "input"]), np.array([1.0 + 1.0j, 2.0])],
)
def test_plot_score_decomposition_rejects_non_real_numeric_components(
    monkeypatch, field, label, value
):
    monkeypatch.setattr(visualization, "_require_matplotlib", lambda: (None, None))

    decomposed = SimpleNamespace(
        combined=np.array([0.1, 0.2]),
        goodness_deficit=None,
        prototype_distance=None,
        transition_surprise=None,
    )
    setattr(decomposed, field, value)

    with pytest.raises(ValueError, match=f"{label} must contain real numeric values"):
        visualization.plot_score_decomposition(decomposed, ax=_Axes())
