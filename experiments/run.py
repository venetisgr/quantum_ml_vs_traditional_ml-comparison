"""Config-driven experiment runner (groundwork §3, §8).

Usage:
    python experiments/run.py --config experiments/configs/e0_smoke.yaml

Contract (fairness protocol):
- Splits are READ from data/splits/<dataset>.json. This runner cannot create
  splits and errors out if the file is missing — never re-split ad hoc.
- The [0, π] AngleScaler is fit on each training fold only.
- One CSV row is appended per (seed, fold) to <results-dir>/<experiment>.csv.
  Results files are append-only: existing rows are never rewritten, and a
  header that differs from this runner's schema aborts the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/run.py` from anywhere
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from data import load_dataset
from data.make_splits import checksum_of
from data.preprocessing import AngleScaler
from models import get_spec

DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_SPLITS_DIR = ROOT / "data" / "splits"

# Simulation regimes are never mixed in one analysis table (CLAUDE.md, §8).
# 'classical' marks models with no quantum component at all.
REGIMES = ("exact", "shots", "noisy", "hardware", "classical")

CSV_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "experiment",
    "dataset",
    "model",
    "config_hash",
    "seed",
    "split_id",
    "regime",
    "n_train",
    "n_test",
    "n_features",
    "n_params",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "train_time_s",
    "eval_time_s",
    "git_commit",
]

REQUIRED_KEYS = {"experiment", "dataset", "model", "regime", "seeds", "folds"}
ALLOWED_KEYS = REQUIRED_KEYS | {"model_params", "description"}


def load_config(path: Path) -> dict:
    with open(path) as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    missing = REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    unknown = config.keys() - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} (typo protection)")
    if config["regime"] not in REGIMES:
        raise ValueError(f"{path}: regime must be one of {REGIMES}, got {config['regime']!r}")
    for key in ("seeds", "folds"):
        value = config[key]
        if (not isinstance(value, list) or not value
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
            raise ValueError(f"{path}: '{key}' must be a non-empty list of ints")
        if len(set(value)) != len(value):
            raise ValueError(f"{path}: '{key}' contains duplicates")
    if config.get("model_params") is not None and not isinstance(config["model_params"], dict):
        raise ValueError(f"{path}: 'model_params' must be a mapping")
    return config


def config_hash(config: dict) -> str:
    """First 10 hex chars of sha256 over the canonicalized config."""
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:10]


def load_splits(splits_dir: Path, dataset: str, expected_n: int) -> dict:
    path = Path(splits_dir) / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No stored splits at {path}. Splits are generated ONCE by "
            f"data/make_splits.py and committed; this runner never re-splits. "
            f"A missing file usually means a typo'd dataset name or a new "
            f"dataset whose splits were not committed yet."
        )
    payload = json.loads(path.read_text())
    if payload["checksum"] != checksum_of(payload["splits"]):
        raise RuntimeError(f"{path}: stored checksum does not match stored indices "
                           "(file corrupted or hand-edited?)")
    stored_n = payload["loader_meta"]["n_samples"]
    if stored_n != expected_n:
        raise RuntimeError(
            f"{path}: splits cover n={stored_n} samples but the loader now returns "
            f"n={expected_n}. The loader drifted; fix the loader — regenerating "
            f"splits is forbidden."
        )
    return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # keep the runner usable in a stripped-down env
        pass


def _positive_class_scores(model, X: np.ndarray) -> np.ndarray | None:
    """Continuous scores for ROC-AUC; None if the model offers neither API."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return None


def git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def append_row(csv_path: Path, row: dict) -> None:
    """Append one result row; never rewrite existing content (CLAUDE.md rule)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    if exists:
        with open(csv_path, newline="") as fh:
            header = next(csv.reader(fh), None)
        if header != CSV_COLUMNS:
            raise RuntimeError(
                f"{csv_path} has a different column set than this runner writes; "
                f"refusing to append. Results files are append-only and "
                f"schema-stable — use a new experiment name instead."
            )
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(
    config_path: Path,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> Path:
    config = load_config(Path(config_path))
    cfg_hash = config_hash(config)
    dataset_name = config["dataset"]
    model_name = config["model"]

    X, y, _meta = load_dataset(dataset_name)
    payload = load_splits(splits_dir, dataset_name, expected_n=len(y))

    bad_seeds = set(config["seeds"]) - set(payload["seeds"])
    if bad_seeds:
        raise ValueError(f"seeds {sorted(bad_seeds)} not in stored splits "
                         f"(available: {payload['seeds']})")
    bad_folds = set(config["folds"]) - set(range(payload["n_folds"]))
    if bad_folds:
        raise ValueError(f"folds {sorted(bad_folds)} out of range "
                         f"0..{payload['n_folds'] - 1}")

    spec = get_spec(model_name)
    commit = git_commit_hash()
    csv_path = Path(results_dir) / f"{config['experiment']}.csv"

    n_rows = 0
    for seed in config["seeds"]:
        for fold in config["folds"]:
            sid = f"s{seed}_f{fold}"
            entry = payload["splits"][sid]
            train_idx = np.asarray(entry["train"], dtype=np.intp)
            test_idx = np.asarray(entry["test"], dtype=np.intp)

            seed_everything(seed)
            # Fairness protocol: scaler fit on the training fold only.
            scaler = AngleScaler().fit(X[train_idx])
            X_train, y_train = scaler.transform(X[train_idx]), y[train_idx]
            X_test, y_test = scaler.transform(X[test_idx]), y[test_idx]

            model = spec.build(dict(config.get("model_params") or {}), seed)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            scores = _positive_class_scores(model, X_test)
            eval_time = time.perf_counter() - t0

            n_params = spec.n_params(model)
            row = {
                "run_id": uuid.uuid4().hex[:12],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "experiment": config["experiment"],
                "dataset": dataset_name,
                "model": model_name,
                "config_hash": cfg_hash,
                "seed": seed,
                "split_id": sid,
                "regime": config["regime"],
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "n_features": int(X.shape[1]),
                "n_params": "" if n_params is None else int(n_params),
                "accuracy": f"{accuracy_score(y_test, y_pred):.6f}",
                "balanced_accuracy": f"{balanced_accuracy_score(y_test, y_pred):.6f}",
                "f1": f"{f1_score(y_test, y_pred, pos_label=1):.6f}",
                "roc_auc": "" if scores is None else f"{roc_auc_score(y_test, scores):.6f}",
                "train_time_s": f"{train_time:.4f}",
                "eval_time_s": f"{eval_time:.4f}",
                "git_commit": commit,
            }
            append_row(csv_path, row)
            n_rows += 1
            print(f"[{config['experiment']}] {dataset_name}/{model_name} {sid}: "
                  f"acc={row['accuracy']} f1={row['f1']}")

    print(f"[{config['experiment']}] appended {n_rows} rows -> {csv_path}")
    return csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven experiment runner (see docs/groundwork.md §3/§8)."
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="YAML config, e.g. experiments/configs/e0_smoke.yaml")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="directory for the append-only CSV (default: results/)")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                        help="directory holding the stored splits (default: data/splits/)")
    args = parser.parse_args(argv)
    run(args.config, results_dir=args.results_dir, splits_dir=args.splits_dir)


if __name__ == "__main__":
    main()
