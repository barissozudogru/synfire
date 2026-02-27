"""Dataset loaders for benchmarking. Currently supports UCR anomaly archive."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_ucr_anomaly(
    name: str, data_dir: str | Path
) -> tuple[NDArray, NDArray, NDArray, int]:
    """Load a UCR anomaly detection dataset.

    UCR format: single text file where each line is a value.
    The filename or an accompanying metadata file specifies the
    training/test split point and anomaly range.

    Args:
        name: Dataset name (filename without extension).
        data_dir: Directory containing the UCR text files.

    Returns:
        (train_series, test_series, labels, anomaly_start)
        where labels is binary (1 = anomaly) aligned with test_series.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    data_dir = Path(data_dir)
    filepath = data_dir / f"{name}.txt"

    if not filepath.exists():
        raise FileNotFoundError(f"UCR dataset not found: {filepath}")

    # UCR format: first line is metadata (train_end anomaly_start anomaly_end)
    # remaining lines are data values
    with open(filepath) as f:
        lines = f.readlines()

    # Parse metadata from first line
    meta = lines[0].strip().split()
    train_end = int(meta[0])
    anomaly_start = int(meta[1])
    anomaly_end = int(meta[2])

    # Parse data
    values = np.array([float(line.strip()) for line in lines[1:] if line.strip()])

    train_series = values[:train_end]
    test_series = values[train_end:]

    # Build labels for test series
    labels = np.zeros(len(test_series))
    # Adjust anomaly positions relative to test start
    rel_start = max(0, anomaly_start - train_end)
    rel_end = min(len(test_series), anomaly_end - train_end)
    if rel_start < rel_end:
        labels[rel_start:rel_end] = 1

    return train_series, test_series, labels, anomaly_start


def list_ucr_datasets(data_dir: str | Path) -> list[str]:
    """List available UCR anomaly detection datasets.

    Args:
        data_dir: Directory containing the UCR text files.

    Returns:
        Sorted list of dataset names (without .txt extension).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(p.stem for p in data_dir.glob("*.txt"))
