"""Internal helper shared by all dataset loaders."""
from __future__ import annotations

import numpy as np


def finalize(
    X,
    y,
    *,
    name: str,
    source: str,
    params: dict,
    label_map: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validate and package a loaded dataset as (X, y, meta).

    Enforces the loader contract everything downstream relies on:
    float64 2-D features, finite values, and labels that are exactly {0, 1}
    (the core task is binary classification, groundwork §5/§11).
    `params` must be JSON-native (lists, not tuples) so that the copy stored
    inside data/splits/<name>.json compares equal after a JSON round trip.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: malformed arrays (X {X.shape}, y {y.shape})")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: X contains NaN or inf")
    classes, counts = np.unique(y, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"{name}: labels must be exactly {{0, 1}}, got {classes.tolist()}")
    meta = {
        "name": name,
        "source": source,
        "params": params,
        "n_samples": int(y.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "label_map": label_map,
    }
    return X, y, meta
