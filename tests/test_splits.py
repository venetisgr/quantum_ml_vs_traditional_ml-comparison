"""Stored-splits tests — the backbone of the fairness protocol (§8).

The committed data/splits/*.json files are THE splits, forever. These tests
prove they are complete, disjoint, stratified, checksummed, in sync with the
loaders, and still exactly reproducible from source.
"""
import json

import numpy as np
import pytest

from data import DATASETS, load_dataset
from data.make_splits import (
    N_FOLDS,
    SEEDS,
    SPLITS_DIR,
    checksum_of,
    make_splits_payload,
    split_id,
)

ALL_DATASETS = sorted(DATASETS)


def _stored(name):
    path = SPLITS_DIR / f"{name}.json"
    assert path.exists(), f"missing committed splits file {path}"
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_stored_file_structure(name):
    payload = _stored(name)
    assert payload["dataset"] == name
    assert payload["seeds"] == list(SEEDS)
    assert payload["n_folds"] == N_FOLDS
    expected_ids = {split_id(s, f) for s in SEEDS for f in range(N_FOLDS)}
    assert set(payload["splits"]) == expected_ids
    assert len(payload["splits"]) == 25


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_checksum_and_loader_meta(name):
    payload = _stored(name)
    assert payload["checksum"] == checksum_of(payload["splits"])
    _, _, meta = load_dataset(name)
    assert payload["loader_meta"] == meta, (
        "loader output drifted from the committed splits — fix the loader; "
        "regenerating splits is forbidden"
    )


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_folds_partition_the_dataset(name):
    payload = _stored(name)
    n = payload["loader_meta"]["n_samples"]
    all_indices = set(range(n))
    for seed in SEEDS:
        covered_test = []
        for fold in range(N_FOLDS):
            entry = payload["splits"][split_id(seed, fold)]
            train, test = set(entry["train"]), set(entry["test"])
            assert train.isdisjoint(test)
            assert train | test == all_indices
            covered_test.extend(entry["test"])
        # across one seed, the 5 test folds partition the dataset exactly
        assert sorted(covered_test) == list(range(n))


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_folds_are_stratified(name):
    payload = _stored(name)
    _, y, _ = load_dataset(name)
    for cls in (0, 1):
        total = int((y == cls).sum())
        lo, hi = total // N_FOLDS, -(-total // N_FOLDS)  # floor, ceil
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                test_idx = np.asarray(payload["splits"][split_id(seed, fold)]["test"])
                count = int((y[test_idx] == cls).sum())
                assert lo <= count <= hi, (
                    f"{name} {split_id(seed, fold)}: class {cls} test count {count} "
                    f"outside stratified range [{lo}, {hi}]"
                )


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_stored_splits_match_regeneration(name):
    stored = _stored(name)
    fresh = make_splits_payload(name)
    assert stored["splits"] == fresh["splits"], (
        f"{name}: committed splits no longer reproducible from source — "
        "environment or loader drifted"
    )
