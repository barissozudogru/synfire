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
