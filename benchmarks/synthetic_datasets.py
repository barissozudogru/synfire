"""Synthetic benchmark datasets with varying signal types and anomaly patterns.

Each generator returns (train_series, test_series, labels) where:
- train_series: clean signal for training
- test_series: signal with injected anomalies
- labels: binary array (1=anomaly) aligned with test_series points
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _inject_anomalies(
    series: NDArray,
    positions: list[int],
    anomaly_type: str = "spike",
    magnitude: float = 5.0,
    width: int = 3,
) -> NDArray:
    """Inject anomalies into a series at given positions."""
    out = series.copy()
    for pos in positions:
        end = min(pos + width, len(out))
        if anomaly_type == "spike":
            out[pos:end] += magnitude
        elif anomaly_type == "dip":
            out[pos:end] -= magnitude
        elif anomaly_type == "noise":
            rng = np.random.default_rng(pos)
            out[pos:end] += rng.standard_normal(end - pos) * magnitude
        elif anomaly_type == "shift":
            out[pos:end] += magnitude
            if end < len(out):
                out[end:] += magnitude * 0.3  # partial mean shift
    return out


def _build_labels(length: int, positions: list[int], radius: int = 20) -> NDArray:
    """Build point-level binary labels around anomaly positions."""
    labels = np.zeros(length)
    for pos in positions:
        start = max(0, pos - radius)
        end = min(length, pos + radius + 1)
        labels[start:end] = 1
    return labels


def sine_spike(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Sine wave with spike anomalies. Easy difficulty."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(1000, dtype=np.float64)
    t_test = np.arange(2000, dtype=np.float64)

    train = np.sin(2 * np.pi * t_train / 50)
    test_clean = np.sin(2 * np.pi * t_test / 50)

    positions = sorted(rng.choice(range(200, 1800, 50), 8, replace=False))
    test = _inject_anomalies(test_clean, positions, "spike", magnitude=5.0)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def sine_noise(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Sine wave with noise burst anomalies. Medium difficulty."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(1000, dtype=np.float64)
    t_test = np.arange(2000, dtype=np.float64)

    train = np.sin(2 * np.pi * t_train / 50) + rng.standard_normal(1000) * 0.1
    rng2 = np.random.default_rng(seed + 100)
    test_clean = np.sin(2 * np.pi * t_test / 50) + rng2.standard_normal(2000) * 0.1

    positions = sorted(rng.choice(range(200, 1800, 60), 6, replace=False))
    test = _inject_anomalies(test_clean, positions, "noise", magnitude=3.0, width=5)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def multi_frequency(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Sum of two sine waves with spike anomalies. Medium difficulty."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(1500, dtype=np.float64)
    t_test = np.arange(2500, dtype=np.float64)

    train = np.sin(2 * np.pi * t_train / 50) + 0.5 * np.sin(2 * np.pi * t_train / 120)
    test_clean = np.sin(2 * np.pi * t_test / 50) + 0.5 * np.sin(2 * np.pi * t_test / 120)

    positions = sorted(rng.choice(range(300, 2200, 70), 8, replace=False))
    test = _inject_anomalies(test_clean, positions, "spike", magnitude=4.0)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def square_wave(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Square wave signal with dip anomalies. Medium difficulty."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(1000, dtype=np.float64)
    t_test = np.arange(2000, dtype=np.float64)

    train = np.sign(np.sin(2 * np.pi * t_train / 80))
    test_clean = np.sign(np.sin(2 * np.pi * t_test / 80))

    positions = sorted(rng.choice(range(200, 1800, 80), 6, replace=False))
    test = _inject_anomalies(test_clean, positions, "dip", magnitude=3.0, width=4)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def sawtooth(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Sawtooth wave with spike anomalies. Medium difficulty."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(1200, dtype=np.float64)
    t_test = np.arange(2400, dtype=np.float64)

    period = 60
    train = 2 * (t_train % period) / period - 1
    test_clean = 2 * (t_test % period) / period - 1

    positions = sorted(rng.choice(range(250, 2100, 65), 7, replace=False))
    test = _inject_anomalies(test_clean, positions, "spike", magnitude=4.0)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def noisy_sine_shift(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Noisy sine with mean shift anomalies. Hard difficulty."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(1000, dtype=np.float64)
    t_test = np.arange(2000, dtype=np.float64)

    train = np.sin(2 * np.pi * t_train / 50) + rng.standard_normal(1000) * 0.2
    rng2 = np.random.default_rng(seed + 200)
    test_clean = np.sin(2 * np.pi * t_test / 50) + rng2.standard_normal(2000) * 0.2

    positions = sorted(rng.choice(range(300, 1700, 100), 5, replace=False))
    test = _inject_anomalies(test_clean, positions, "shift", magnitude=2.0, width=8)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def random_walk(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """Random walk with spike anomalies. Hard difficulty."""
    rng = np.random.default_rng(seed)

    steps_train = rng.standard_normal(1500) * 0.1
    train = np.cumsum(steps_train)
    train = train - np.polyval(np.polyfit(np.arange(len(train)), train, 1), np.arange(len(train)))

    rng2 = np.random.default_rng(seed + 300)
    steps_test = rng2.standard_normal(2500) * 0.1
    test_clean = np.cumsum(steps_test)
    test_clean = test_clean - np.polyval(
        np.polyfit(np.arange(len(test_clean)), test_clean, 1), np.arange(len(test_clean))
    )

    positions = sorted(rng.choice(range(300, 2200, 80), 7, replace=False))
    test = _inject_anomalies(test_clean, positions, "spike", magnitude=3.0)
    labels = _build_labels(len(test), positions)

    return train, test, labels


def ecg_like(seed: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """ECG-like signal (sum of Gaussians) with anomalies. Hard difficulty."""
    rng = np.random.default_rng(seed)

    def _ecg_cycle(t: NDArray, period: int = 100) -> NDArray:
        phase = (t % period) / period
        # P wave
        p = 0.2 * np.exp(-((phase - 0.15) ** 2) / 0.002)
        # QRS complex
        qrs = 1.0 * np.exp(-((phase - 0.35) ** 2) / 0.0008)
        qrs -= 0.3 * np.exp(-((phase - 0.30) ** 2) / 0.001)
        # T wave
        t_wave = 0.3 * np.exp(-((phase - 0.55) ** 2) / 0.004)
        return p + qrs + t_wave

    t_train = np.arange(1500, dtype=np.float64)
    t_test = np.arange(2500, dtype=np.float64)

    train = _ecg_cycle(t_train) + rng.standard_normal(1500) * 0.05
    rng2 = np.random.default_rng(seed + 400)
    test_clean = _ecg_cycle(t_test) + rng2.standard_normal(2500) * 0.05

    positions = sorted(rng.choice(range(300, 2200, 100), 6, replace=False))
    test = _inject_anomalies(test_clean, positions, "spike", magnitude=2.0)
    labels = _build_labels(len(test), positions)

    return train, test, labels


ALL_DATASETS: dict[str, callable] = {
    "sine_spike": sine_spike,
    "sine_noise": sine_noise,
    "multi_frequency": multi_frequency,
    "square_wave": square_wave,
    "sawtooth": sawtooth,
    "noisy_sine_shift": noisy_sine_shift,
    "random_walk": random_walk,
    "ecg_like": ecg_like,
}
