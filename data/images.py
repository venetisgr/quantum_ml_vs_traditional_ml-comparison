"""Reduced image datasets, Tier 2 (groundwork §5).

E0 ships the sklearn digits 3-vs-5 pair at full 64-dimensional (8×8 flattened)
resolution; the PCA-8/16 reduction for quantum models is a config-level
preprocessing step in later experiments, not part of the loader. MNIST /
Fashion-MNIST binary pairs are added additively when the image tier starts.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits

from data._common import finalize


def load_digits_3v5() -> tuple[np.ndarray, np.ndarray, dict]:
    """sklearn 8×8 digits restricted to the visually confusable 3-vs-5 pair."""
    bunch = load_digits()
    mask = np.isin(bunch.target, (3, 5))
    X = bunch.data[mask]  # (n, 64), pixel values in [0, 16]
    y = (bunch.target[mask] == 5).astype(np.int64)
    return finalize(
        X,
        y,
        name="digits_3v5",
        source="sklearn.datasets.load_digits",
        params={"classes": [3, 5]},
        label_map={"0": "digit 3", "1": "digit 5"},
    )
