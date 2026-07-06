"""Small tabular Tier-1 datasets (groundwork §5), binarized where needed.

Every target is remapped to {0, 1}; `label_map` in each loader's meta records
the original classes. Banknote is fetched from OpenML on first use and cached
by scikit-learn (no raw data in git, groundwork §3). Feature counts above the
~20-qubit budget (breast_cancer: 30) are returned at full dimensionality —
PCA reduction for quantum models is a config-level step in later experiments.
"""
from __future__ import annotations

import numpy as np
from sklearn import datasets as _sk_datasets

from data._common import finalize

# The hardest wine class pair, chosen by the deterministic probe below
# (`python -m data.tabular`, run 2026-07-06): mean 5-fold CV logreg accuracy
# (1,2)=0.9917 < (0,1)=0.9923 < (0,2)=1.0000. Near-tie with (0,1); decision
# logged in STATUS.md. None would mean "probe not run yet" — loaders and split
# generation refuse to proceed in that state.
WINE_BINARY_CLASSES: tuple[int, int] | None = (1, 2)


def load_iris_binary() -> tuple[np.ndarray, np.ndarray, dict]:
    """Iris restricted to versicolor vs virginica — the non-linearly-separable pair."""
    bunch = _sk_datasets.load_iris()
    mask = np.isin(bunch.target, (1, 2))
    X = bunch.data[mask]
    y = (bunch.target[mask] == 2).astype(np.int64)
    return finalize(
        X,
        y,
        name="iris_binary",
        source="sklearn.datasets.load_iris",
        params={"classes": [1, 2]},
        label_map={"0": "versicolor", "1": "virginica"},
    )


def load_wine_binary() -> tuple[np.ndarray, np.ndarray, dict]:
    """Wine restricted to its hardest class pair (see WINE_BINARY_CLASSES)."""
    if WINE_BINARY_CLASSES is None:
        raise RuntimeError(
            "WINE_BINARY_CLASSES is undecided — run `python -m data.tabular` and "
            "set the constant to the probe's hardest pair before using wine_binary."
        )
    a, b = WINE_BINARY_CLASSES
    bunch = _sk_datasets.load_wine()
    mask = np.isin(bunch.target, (a, b))
    X = bunch.data[mask]
    y = (bunch.target[mask] == b).astype(np.int64)
    return finalize(
        X,
        y,
        name="wine_binary",
        source="sklearn.datasets.load_wine",
        params={"classes": [a, b]},
        label_map={"0": f"class_{a}", "1": f"class_{b}"},
    )


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray, dict]:
    """Breast Cancer Wisconsin, already binary (0 = malignant, 1 = benign)."""
    bunch = _sk_datasets.load_breast_cancer()
    return finalize(
        bunch.data,
        bunch.target,
        name="breast_cancer",
        source="sklearn.datasets.load_breast_cancer",
        params={},
        label_map={"0": "malignant", "1": "benign"},
    )


def load_banknote() -> tuple[np.ndarray, np.ndarray, dict]:
    """Banknote authentication (OpenML data_id 1462), fetched once then cached."""
    bunch = _sk_datasets.fetch_openml(data_id=1462, as_frame=False)
    X = bunch.data
    y = (bunch.target == "2").astype(np.int64)  # OpenML target values are '1'/'2'
    return finalize(
        X,
        y,
        name="banknote",
        source="sklearn.datasets.fetch_openml",
        params={"data_id": 1462},
        label_map={"0": "genuine (openml target '1')", "1": "forged (openml target '2')"},
    )


def load_pima() -> tuple[np.ndarray, np.ndarray, dict]:
    """Pima Indians Diabetes (OpenML data_id 37) — hard tier, E1 extension.

    768×8, mildly imbalanced (500/268), noisy clinical measurements; tuned
    classical models plateau around 0.75 balanced accuracy in the literature.
    """
    bunch = _sk_datasets.fetch_openml(data_id=37, as_frame=False)
    y = (bunch.target == "tested_positive").astype(np.int64)
    return finalize(
        bunch.data,
        y,
        name="pima",
        source="sklearn.datasets.fetch_openml",
        params={"data_id": 37},
        label_map={"0": "tested_negative", "1": "tested_positive"},
    )


def load_heart_statlog() -> tuple[np.ndarray, np.ndarray, dict]:
    """Heart disease, Statlog variant of Cleveland (OpenML data_id 53) — hard tier.

    270×13, no missing values (the cleaned Cleveland release groundwork §5
    refers to); typical tuned accuracy ~0.83.
    """
    bunch = _sk_datasets.fetch_openml(data_id=53, as_frame=False)
    y = (bunch.target == "present").astype(np.int64)
    return finalize(
        bunch.data,
        y,
        name="heart_statlog",
        source="sklearn.datasets.fetch_openml",
        params={"data_id": 53},
        label_map={"0": "absent", "1": "present"},
    )


def wine_pair_probe(n_splits: int = 5, seed: int = 0) -> dict[tuple[int, int], float]:
    """Difficulty probe for the wine pair decision (run: python -m data.tabular).

    Standardized logistic regression, stratified 5-fold CV accuracy per class
    pair; the LOWEST-accuracy pair is declared hardest and becomes
    WINE_BINARY_CLASSES. Deliberately a simple, fixed probe — it only has to
    rank the pairs deterministically, not benchmark anything.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    bunch = _sk_datasets.load_wine()
    results: dict[tuple[int, int], float] = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        mask = np.isin(bunch.target, (a, b))
        X = bunch.data[mask]
        y = (bunch.target[mask] == b).astype(np.int64)
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        results[(a, b)] = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
    return results


if __name__ == "__main__":
    probe = wine_pair_probe()
    for pair, acc in sorted(probe.items(), key=lambda kv: kv[1]):
        print(f"classes {pair}: mean CV accuracy {acc:.4f}")
    hardest = min(probe, key=probe.get)
    print(f"hardest pair -> WINE_BINARY_CLASSES = {hardest}")
