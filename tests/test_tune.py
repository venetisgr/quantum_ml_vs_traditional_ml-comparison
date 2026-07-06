"""Tuning-protocol tests (§8): study determinism, locked configs, log format."""
import csv
import json

import pytest
import yaml

from experiments.run import load_config, run
from experiments.tune import (
    TRIALS_COLUMNS,
    TUNING_SPLIT_ID,
    inner_cv_score,
    tune_one,
    write_default_config,
)
from models import get_spec


def _tune(tmp_path, model="logreg", dataset="moons", n_trials=3, tag="a"):
    return tune_one(
        model, dataset, n_trials=n_trials,
        trials_log=tmp_path / f"trials_{tag}.csv",
        locked_dir=tmp_path / f"locked_{tag}",
    )


def test_tune_one_writes_valid_locked_config(tmp_path):
    path = _tune(tmp_path)
    config = load_config(path)  # passes the runner's own validation
    assert config["experiment"] == "e1_baselines"
    assert config["dataset"] == "moons" and config["model"] == "logreg"
    assert config["regime"] == "classical"
    assert config["seeds"] == [0, 1, 2, 3, 4] and config["folds"] == [0, 1, 2, 3, 4]
    assert 1e-3 <= config["model_params"]["C"] <= 1e3
    assert TUNING_SPLIT_ID in config["description"]


def test_trials_log_schema_and_count(tmp_path):
    _tune(tmp_path, tag="log")
    with open(tmp_path / "trials_log.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == TRIALS_COLUMNS
        rows = list(reader)
    assert len(rows) == 3
    for row in rows:
        assert row["objective"] == "balanced_accuracy"
        assert 0.0 <= float(row["value"]) <= 1.0
        json.loads(row["params_json"])  # valid JSON


def test_studies_are_deterministic(tmp_path):
    p1 = _tune(tmp_path, model="svm_rbf", tag="d1")
    p2 = _tune(tmp_path, model="svm_rbf", tag="d2")
    c1, c2 = yaml.safe_load(p1.read_text()), yaml.safe_load(p2.read_text())
    assert c1["model_params"] == c2["model_params"]


def test_existing_locked_config_is_not_overwritten(tmp_path):
    path = _tune(tmp_path, tag="skip")
    before = path.read_text()
    path.write_text(before + "# sentinel\n")
    again = tune_one("logreg", "moons", n_trials=3,
                     trials_log=tmp_path / "trials_skip2.csv",
                     locked_dir=tmp_path / "locked_skip")
    assert again == path
    assert path.read_text().endswith("# sentinel\n")  # untouched without --force


def test_locked_config_runs_through_the_runner(tmp_path):
    path = _tune(tmp_path, tag="run")
    config = yaml.safe_load(path.read_text())
    config["seeds"], config["folds"] = [0], [0]  # single fold for speed
    small = tmp_path / "small.yaml"
    small.write_text(yaml.safe_dump(config))
    csv_path = run(small, results_dir=tmp_path / "results")
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1 and rows[0]["experiment"] == "e1_baselines"


def test_inner_cv_scores_are_sane():
    import numpy as np
    from data import load_dataset

    X, y, _ = load_dataset("moons")
    spec = get_spec("logreg")
    score = inner_cv_score(spec, {"C": 1.0}, X[:400], y[:400])
    assert 0.7 <= score <= 1.0


def test_write_default_config(tmp_path, monkeypatch):
    import experiments.tune as tune_mod

    monkeypatch.setattr(tune_mod, "DEFAULTS_DIR", tmp_path / "defaults")
    path = write_default_config("mlp", "moons")
    config = load_config(path)
    assert config["experiment"] == "e1_baselines_default"
    assert config["model_params"] == {}
