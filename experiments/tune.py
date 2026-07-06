"""E1 hyperparameter tuning — the §8 matched-budget protocol, operationalized.

Per (model, dataset): ONE Optuna study (default 50 trials, TPESampler(seed=0));
every trial is scored by 3-fold stratified CV on the TRAINING portion of the
stored split s0_f0 only — no test fold is ever touched during tuning. The best
config is locked to experiments/configs/e1_locked/<name>.yaml and later
evaluated on all 25 (seed, fold) pairs by experiments/run.py. The procedure
and budget are identical for every model — classical now, quantum in E2+.
Residual train-fold overlap (non-nested tuning) is a stated Ch. 6 limitation.

Objective: balanced accuracy (decision 2026-07-06, STATUS.md).
Every trial is appended to results/e1_tuning_trials.csv (thesis appendix).

Usage:
  python experiments/tune.py --all [--trials 50]
  python experiments/tune.py --models mlp xgboost --datasets moons banknote
  python experiments/tune.py --write-defaults    # default-variant configs only

Locked configs are never overwritten without --force (tuning is part of the
frozen protocol once run).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from data import available_datasets, load_dataset
from data.preprocessing import AngleScaler
from experiments.run import DEFAULT_SPLITS_DIR, load_splits
from models import SPECS, get_spec

TUNING_SPLIT_ID = "s0_f0"   # §8: training portion of seed-0/fold-0
INNER_FOLDS = 3
INNER_SEED = 0
STUDY_SEED = 0
DEFAULT_TRIALS = 50
OBJECTIVE = "balanced_accuracy"

LOCKED_DIR = ROOT / "experiments" / "configs" / "e1_locked"
DEFAULTS_DIR = ROOT / "experiments" / "configs" / "e1_defaults"
TRIALS_LOG = ROOT / "results" / "e1_tuning_trials.csv"

TRIALS_COLUMNS = [
    "timestamp_utc", "model", "dataset", "sampler", "study_seed",
    "trial", "objective", "value", "params_json",
]

EVAL_SEEDS = [0, 1, 2, 3, 4]
EVAL_FOLDS = [0, 1, 2, 3, 4]

# MLPs at tiny widths legitimately hit max_iter during search; the warning
# noise would drown the log without changing any result.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _append_row(csv_path: Path, columns: list[str], row: dict) -> None:
    """Append-only CSV writer with header-drift protection (results/ rules)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    if exists:
        with open(csv_path, newline="") as fh:
            header = next(csv.reader(fh), None)
        if header != columns:
            raise RuntimeError(f"{csv_path}: existing header differs; refusing to append")
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def inner_cv_score(spec, params: dict, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Mean balanced accuracy over 3 stratified inner folds of the tuning split.

    Mirrors the runner exactly: the [0, π] scaler is fit on each inner
    training portion only.
    """
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    scores = []
    for inner_train, inner_val in skf.split(X_train, y_train):
        scaler = AngleScaler().fit(X_train[inner_train])
        model = spec.build(dict(params), INNER_SEED)
        model.fit(scaler.transform(X_train[inner_train]), y_train[inner_train])
        pred = model.predict(scaler.transform(X_train[inner_val]))
        scores.append(balanced_accuracy_score(y_train[inner_val], pred))
    return float(np.mean(scores))


def locked_config_path(dataset: str, model_name: str) -> Path:
    return LOCKED_DIR / f"e1__{dataset}__{model_name}.yaml"


def default_config_path(dataset: str, model_name: str) -> Path:
    return DEFAULTS_DIR / f"e1__{dataset}__{model_name}.yaml"


def tune_one(
    model_name: str,
    dataset: str,
    n_trials: int = DEFAULT_TRIALS,
    trials_log: Path = TRIALS_LOG,
    locked_dir: Path | None = None,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    force: bool = False,
) -> Path:
    """Run one §8 study and write the locked config; returns the config path."""
    out_path = (locked_dir or LOCKED_DIR) / f"e1__{dataset}__{model_name}.yaml"
    if out_path.exists() and not force:
        print(f"{dataset}/{model_name}: {out_path.name} exists — skipping (--force to redo)")
        return out_path

    spec = get_spec(model_name)
    if spec.suggest is None:
        raise ValueError(f"model '{model_name}' has no search space")

    X, y, _ = load_dataset(dataset)
    payload = load_splits(splits_dir, dataset, expected_n=len(y))
    train_idx = np.asarray(payload["splits"][TUNING_SPLIT_ID]["train"], dtype=np.intp)
    X_train, y_train = X[train_idx], y[train_idx]

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest(trial)
        trial.set_user_attr("params", params)
        return inner_cv_score(spec, params, X_train, y_train)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=STUDY_SEED),
        study_name=f"e1__{dataset}__{model_name}",
    )
    study.optimize(objective, n_trials=n_trials)

    for t in study.trials:
        _append_row(trials_log, TRIALS_COLUMNS, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model_name,
            "dataset": dataset,
            "sampler": "TPE",
            "study_seed": STUDY_SEED,
            "trial": t.number,
            "objective": OBJECTIVE,
            "value": f"{t.value:.6f}",
            "params_json": json.dumps(t.user_attrs["params"], sort_keys=True),
        })

    best = study.best_trial
    best_params = best.user_attrs["params"]
    config = {
        "experiment": "e1_baselines",
        "description": (
            f"E1 locked config for {model_name} on {dataset}: best of {n_trials} "
            f"TPE(seed={STUDY_SEED}) trials, {INNER_FOLDS}-fold CV on the training "
            f"portion of {TUNING_SPLIT_ID}, objective {OBJECTIVE}={best.value:.4f} "
            f"(trial {best.number}). Generated by experiments/tune.py — do not edit."
        ),
        "dataset": dataset,
        "model": model_name,
        "model_params": best_params,
        "regime": "classical",
        "seeds": EVAL_SEEDS,
        "folds": EVAL_FOLDS,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"{dataset}/{model_name}: best {OBJECTIVE}={best.value:.4f} "
          f"(trial {best.number}) -> {out_path.name}")
    return out_path


def write_default_config(model_name: str, dataset: str) -> Path:
    """Default-variant config: build() defaults, same 25-fold evaluation."""
    config = {
        "experiment": "e1_baselines_default",
        "description": (
            f"E1 default (untuned) variant of {model_name} on {dataset}; "
            f"build() defaults, evaluated on the same 25 stored folds. "
            f"Generated by experiments/tune.py — do not edit."
        ),
        "dataset": dataset,
        "model": model_name,
        "model_params": {},
        "regime": "classical",
        "seeds": EVAL_SEEDS,
        "folds": EVAL_FOLDS,
    }
    path = default_config_path(dataset, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", choices=sorted(SPECS), default=None)
    parser.add_argument("--datasets", nargs="+", choices=available_datasets(), default=None)
    parser.add_argument("--all", action="store_true",
                        help="tune every model on every dataset")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--force", action="store_true",
                        help="re-tune and overwrite an existing locked config")
    parser.add_argument("--write-defaults", action="store_true",
                        help="write the default-variant configs for the grid and exit")
    args = parser.parse_args(argv)

    models = args.models or (sorted(SPECS) if (args.all or args.write_defaults) else None)
    datasets = args.datasets or (available_datasets() if (args.all or args.write_defaults) else None)
    if not models or not datasets:
        parser.error("pass --all, or both --models and --datasets")

    if args.write_defaults:
        for dataset in datasets:
            for model_name in models:
                write_default_config(model_name, dataset)
        print(f"wrote {len(models) * len(datasets)} default configs -> {DEFAULTS_DIR}")
        return

    total = len(models) * len(datasets)
    done = 0
    for dataset in datasets:
        for model_name in models:
            done += 1
            print(f"[{done}/{total}] tuning {model_name} on {dataset} "
                  f"({args.trials} trials)...", flush=True)
            tune_one(model_name, dataset, n_trials=args.trials, force=args.force)


if __name__ == "__main__":
    main()
