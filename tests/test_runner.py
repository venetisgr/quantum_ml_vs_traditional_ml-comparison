"""End-to-end runner tests on the committed e0_smoke.yaml config (moons + logreg)."""
import csv
from pathlib import Path

import pytest
import yaml

from experiments.run import (
    CSV_COLUMNS,
    ROOT,
    config_hash,
    load_config,
    run,
)

SMOKE_CONFIG = ROOT / "experiments" / "configs" / "e0_smoke.yaml"


def _read_rows(csv_path: Path):
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_COLUMNS
        return list(reader)


def test_end_to_end_smoke(tmp_path):
    csv_path = run(SMOKE_CONFIG, results_dir=tmp_path)
    assert csv_path == tmp_path / "e0_smoke.csv"
    rows = _read_rows(csv_path)
    assert len(rows) == 25  # 5 seeds x 5 folds

    assert {r["dataset"] for r in rows} == {"moons"}
    assert {r["model"] for r in rows} == {"logreg"}
    assert {r["regime"] for r in rows} == {"classical"}
    assert {r["split_id"] for r in rows} == {f"s{s}_f{f}" for s in range(5) for f in range(5)}
    assert len({r["config_hash"] for r in rows}) == 1
    # logreg on 2 features: 2 coefficients + 1 intercept, logged for every row
    assert {r["n_params"] for r in rows} == {"3"}
    assert {r["n_train"] for r in rows} == {"400"}
    assert {r["n_test"] for r in rows} == {"100"}

    accuracies = [float(r["accuracy"]) for r in rows]
    assert sum(accuracies) / len(accuracies) > 0.85
    for r in rows:
        assert 0.0 <= float(r["balanced_accuracy"]) <= 1.0
        assert 0.0 <= float(r["f1"]) <= 1.0
        assert 0.5 < float(r["roc_auc"]) <= 1.0


def test_second_run_appends_and_is_deterministic(tmp_path):
    run(SMOKE_CONFIG, results_dir=tmp_path)
    run(SMOKE_CONFIG, results_dir=tmp_path)
    rows = _read_rows(tmp_path / "e0_smoke.csv")
    assert len(rows) == 50, "second run must append, never overwrite"
    first, second = rows[:25], rows[25:]
    # identical config + stored splits + seeding => identical metrics
    assert [r["accuracy"] for r in first] == [r["accuracy"] for r in second]
    assert [r["config_hash"] for r in first] == [r["config_hash"] for r in second]


def test_foreign_header_is_refused(tmp_path):
    (tmp_path / "e0_smoke.csv").write_text("some,other,columns\n1,2,3\n")
    with pytest.raises(RuntimeError, match="refusing to append"):
        run(SMOKE_CONFIG, results_dir=tmp_path)


def test_missing_splits_file_is_fatal(tmp_path):
    empty_splits_dir = tmp_path / "no_splits_here"
    empty_splits_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="never re-splits"):
        run(SMOKE_CONFIG, results_dir=tmp_path, splits_dir=empty_splits_dir)


def test_config_hash_is_stable_and_canonical():
    config = load_config(SMOKE_CONFIG)
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))
    assert config_hash(config) != config_hash({**config, "model": "svm_rbf"})


@pytest.mark.parametrize(
    "corruption, message",
    [
        ({"regime": "quantumish"}, "regime"),
        ({"unexpected_key": 1}, "unknown keys"),
        ({"seeds": []}, "non-empty list"),
        ({"seeds": [0, 0]}, "duplicates"),
        ({"folds": "all"}, "non-empty list"),
    ],
)
def test_config_validation_rejects(tmp_path, corruption, message):
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    config.update(corruption)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        load_config(bad)


def test_out_of_range_seeds_rejected(tmp_path):
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    config["seeds"] = [0, 7]
    bad = tmp_path / "bad_seeds.yaml"
    bad.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="not in stored splits"):
        run(bad, results_dir=tmp_path)
