"""Synthetic Tier-0 datasets (groundwork §5).

Generation parameters are FIXED here, as the single source of truth, so the
stored splits in data/splits/ always correspond exactly to the arrays these
loaders return. A variant (different n, noise, ...) must become a NEW dataset
name with its own stored splits — never a parameter override.

Deferred Tier-0 generators, added additively when E2 starts: the
qml-benchmarks hidden-manifold family and the Qiskit ad-hoc dataset
(parity/XOR if needed for E4).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons

from data._common import finalize

# Dataset-synthesis seed; deliberately distinct from the split seeds 0-4 and
# from all run seeds (decision logged in STATUS.md, 2026-07-06).
GENERATION_SEED = 1234
N_SAMPLES = 500
MOONS_NOISE = 0.2
CIRCLES_NOISE = 0.1
CIRCLES_FACTOR = 0.5


def load_moons() -> tuple[np.ndarray, np.ndarray, dict]:
    """Two interleaving half-moons; the sanity-check dataset used by E0."""
    X, y = make_moons(n_samples=N_SAMPLES, noise=MOONS_NOISE, random_state=GENERATION_SEED)
    return finalize(
        X,
        y,
        name="moons",
        source="sklearn.datasets.make_moons",
        params={"n_samples": N_SAMPLES, "noise": MOONS_NOISE, "random_state": GENERATION_SEED},
        label_map={"0": "moon 0", "1": "moon 1"},
    )


def load_circles() -> tuple[np.ndarray, np.ndarray, dict]:
    """Concentric circles; radially symmetric — a different boundary family than moons."""
    X, y = make_circles(
        n_samples=N_SAMPLES,
        noise=CIRCLES_NOISE,
        factor=CIRCLES_FACTOR,
        random_state=GENERATION_SEED,
    )
    return finalize(
        X,
        y,
        name="circles",
        source="sklearn.datasets.make_circles",
        params={
            "n_samples": N_SAMPLES,
            "noise": CIRCLES_NOISE,
            "factor": CIRCLES_FACTOR,
            "random_state": GENERATION_SEED,
        },
        label_map={"0": "outer circle", "1": "inner circle"},
    )
