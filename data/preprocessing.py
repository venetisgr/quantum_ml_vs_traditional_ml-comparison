"""Shared preprocessing (fairness protocol, groundwork §5/§8).

Every model — classical and quantum alike — consumes the same per-feature
min–max scaling onto [0, π], the angle-encoding range (matched-conditions
decision logged in STATUS.md, 2026-07-06). The scaler is ALWAYS fit on the
training fold only, inside the runner; fitting on the full dataset would leak
test-fold statistics into training and undermine the comparison.
"""
from __future__ import annotations

import numpy as np

ANGLE_MAX = float(np.pi)


class AngleScaler:
    """Per-feature min–max scaling of the training fold onto [0, π].

    Test-fold values falling outside the training range are clipped to the
    bounds, so encoded angles always stay inside [0, π]. A feature that is
    constant on the training fold maps to 0.0 everywhere.
    """

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "AngleScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("AngleScaler.fit: X contains NaN or inf")
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("AngleScaler.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        safe_range = np.where(self.range_ > 0, self.range_, 1.0)
        scaled = (X - self.min_) / safe_range
        scaled = np.where(self.range_ > 0, scaled, 0.0)
        return np.clip(scaled, 0.0, 1.0) * ANGLE_MAX

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
