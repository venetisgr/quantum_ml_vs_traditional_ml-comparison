"""Loader contract tests for every registered dataset (groundwork §5 core set)."""
import numpy as np
import pytest

from data import DATASETS, available_datasets, load_dataset

ALL_DATASETS = sorted(DATASETS)

# Ground truth (name -> (n_samples, n_features, count_class0, count_class1)),
# recorded when the shared splits were generated on 2026-07-06. Any change here
# means a loader drifted and would invalidate the committed splits.
EXPECTED = {
    "moons": (500, 2, 250, 250),
    "circles": (500, 2, 250, 250),
    "iris_binary": (100, 4, 50, 50),
    "wine_binary": (119, 13, 71, 48),
    "breast_cancer": (569, 30, 212, 357),
    "digits_3v5": (365, 64, 183, 182),
    "banknote": (1372, 4, 762, 610),
}


def test_registry_matches_expected_table():
    assert set(ALL_DATASETS) == set(EXPECTED)


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_loader_contract(name):
    X, y, meta = load_dataset(name)
    assert isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float64
    assert isinstance(y, np.ndarray) and y.dtype == np.int64
    assert y.shape == (X.shape[0],)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == {0, 1}
    assert meta["name"] == name
    assert meta["n_samples"] == X.shape[0]
    assert meta["n_features"] == X.shape[1]
    assert meta["class_counts"]["0"] == int((y == 0).sum())
    assert meta["class_counts"]["1"] == int((y == 1).sum())
    for key in ("source", "params", "label_map"):
        assert key in meta, f"meta missing '{key}'"
    assert set(meta["label_map"]) == {"0", "1"}


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_loader_matches_ground_truth(name):
    X, y, _ = load_dataset(name)
    n, d, c0, c1 = EXPECTED[name]
    assert X.shape == (n, d)
    assert int((y == 0).sum()) == c0
    assert int((y == 1).sum()) == c1


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_loader_deterministic(name):
    X1, y1, m1 = load_dataset(name)
    X2, y2, m2 = load_dataset(name)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert m1 == m2


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        load_dataset("does_not_exist")


def test_available_datasets_sorted():
    assert available_datasets() == ALL_DATASETS
