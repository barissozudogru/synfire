import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sine_series():
    """1000-point sine wave with period 50."""
    t = np.arange(1000, dtype=np.float64)
    return np.sin(2 * np.pi * t / 50)


@pytest.fixture
def multivariate_series(rng):
    """3-channel series of length 500."""
    t = np.arange(500, dtype=np.float64)
    return np.column_stack([
        np.sin(2 * np.pi * t / 50),
        np.cos(2 * np.pi * t / 30),
        rng.standard_normal(500) * 0.1,
    ])


@pytest.fixture
def sine_with_anomalies():
    """Sine wave with injected spike anomalies at known positions."""
    t = np.arange(1000, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 50)
    anomaly_positions = [200, 400, 600, 800]
    for pos in anomaly_positions:
        series[pos : pos + 5] += 5.0
    return series, anomaly_positions
