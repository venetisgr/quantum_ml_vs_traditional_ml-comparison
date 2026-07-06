"""Dataset registry — the E0 core set from groundwork §5.

`load_dataset(name)` is the only entry point the rest of the codebase uses.
Loaders take NO parameters: every generation/binarization choice is fixed
inside the loader itself, so the stored splits in data/splits/ always match
the arrays a loader returns. New datasets (hidden-manifold, Qiskit ad-hoc,
MNIST pairs, ...) are added here additively when their experiment tier starts.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from data.images import load_digits_3v5
from data.synthetic import load_circles, load_moons
from data.tabular import (
    load_banknote,
    load_breast_cancer,
    load_iris_binary,
    load_wine_binary,
)

Loader = Callable[[], tuple[np.ndarray, np.ndarray, dict]]

DATASETS: dict[str, Loader] = {
    "moons": load_moons,
    "circles": load_circles,
    "iris_binary": load_iris_binary,
    "wine_binary": load_wine_binary,
    "breast_cancer": load_breast_cancer,
    "digits_3v5": load_digits_3v5,
    "banknote": load_banknote,
}


def available_datasets() -> list[str]:
    return sorted(DATASETS)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        loader = DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available_datasets()}"
        ) from None
    return loader()
