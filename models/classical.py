"""Classical baseline models (groundwork §6) — the complete E1 zoo.

Seven baselines: logistic regression, SVM-linear, SVM-RBF, k-NN, random
forest, gradient boosting (XGBoost), and a small MLP. Each entry provides:

  build(params, seed)  -> unfitted sklearn-compatible estimator
  n_params(estimator)  -> learned parameter count after fit, or None where the
                          notion does not apply (kernel expansions, trees, k-NN)
  suggest(trial)       -> Optuna search space (≤ 6 hyperparameters per model —
                          the same "size philosophy" quantum models get, §8)

Search spaces live HERE, next to the model they tune, so model and space
cannot drift apart. Determinism knobs (random_state, n_jobs=1, fixed
max_iter) are set in build(), not searched.

MLP policy (decision 2026-07-06): one hidden layer, width searched 2–32,
parameter count logged on every run; strict ≤ ~200-parameter matching against
VQCs is enforced in E3 when VQC sizes exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[dict, int], Any]
    n_params: Callable[[Any], int | None]
    suggest: Callable[[Any], dict] | None = None  # takes an optuna Trial


def _no_params(est: Any) -> None:
    # Support-vector expansions, tree ensembles, and k-NN have no fixed
    # parameter count comparable to NN/VQC weights (§8 parameter matching
    # applies to NN-style models only).
    return None


# --- logistic regression ----------------------------------------------------

def _build_logreg(params: dict, seed: int) -> LogisticRegression:
    defaults: dict[str, Any] = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}
    defaults.update(params)
    return LogisticRegression(random_state=seed, **defaults)


def _logreg_n_params(est: LogisticRegression) -> int:
    return int(est.coef_.size + est.intercept_.size)


def _suggest_logreg(trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 1e3, log=True)}


# --- SVMs -------------------------------------------------------------------

def _build_svm_linear(params: dict, seed: int) -> SVC:
    defaults: dict[str, Any] = {"C": 1.0, "kernel": "linear"}
    defaults.update(params)
    return SVC(random_state=seed, **defaults)


def _svm_linear_n_params(est: SVC) -> int:
    # A linear kernel yields an explicit weight vector.
    return int(est.coef_.size + est.intercept_.size)


def _suggest_svm_linear(trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 1e3, log=True)}


def _build_svm_rbf(params: dict, seed: int) -> SVC:
    defaults: dict[str, Any] = {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    defaults.update(params)
    return SVC(random_state=seed, **defaults)


def _suggest_svm_rbf(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
    }


# --- k-nearest neighbours ---------------------------------------------------

def _build_knn(params: dict, seed: int) -> KNeighborsClassifier:
    defaults: dict[str, Any] = {"n_neighbors": 5, "weights": "uniform", "p": 2}
    defaults.update(params)
    return KNeighborsClassifier(**defaults)  # deterministic; no seed to pass


def _suggest_knn(trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2]),
    }


# --- random forest ----------------------------------------------------------

def _build_rf(params: dict, seed: int) -> RandomForestClassifier:
    defaults: dict[str, Any] = {"n_estimators": 100, "n_jobs": 1}
    defaults.update(params)
    return RandomForestClassifier(random_state=seed, **defaults)


def _suggest_rf(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


# --- gradient boosting (XGBoost) ---------------------------------------------

def _build_xgboost(params: dict, seed: int) -> XGBClassifier:
    defaults: dict[str, Any] = {
        "n_estimators": 100,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": 1,
    }
    defaults.update(params)
    return XGBClassifier(random_state=seed, **defaults)


def _suggest_xgboost(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


# --- small MLP ----------------------------------------------------------------

def _build_mlp(params: dict, seed: int) -> MLPClassifier:
    defaults: dict[str, Any] = {
        "hidden_layer_sizes": [8],   # small-MLP default; tuned variant searches 2-32
        "max_iter": 1000,
        "early_stopping": False,
    }
    defaults.update(params)
    return MLPClassifier(random_state=seed, **defaults)


def _mlp_n_params(est: MLPClassifier) -> int:
    return int(sum(c.size for c in est.coefs_) + sum(i.size for i in est.intercepts_))


def _suggest_mlp(trial) -> dict:
    width = trial.suggest_int("hidden_width", 2, 32)
    return {
        "hidden_layer_sizes": [width],
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
    }


SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", _build_logreg, _logreg_n_params, _suggest_logreg),
    "svm_linear": ModelSpec("svm_linear", _build_svm_linear, _svm_linear_n_params, _suggest_svm_linear),
    "svm_rbf": ModelSpec("svm_rbf", _build_svm_rbf, _no_params, _suggest_svm_rbf),
    "knn": ModelSpec("knn", _build_knn, _no_params, _suggest_knn),
    "rf": ModelSpec("rf", _build_rf, _no_params, _suggest_rf),
    "xgboost": ModelSpec("xgboost", _build_xgboost, _no_params, _suggest_xgboost),
    "mlp": ModelSpec("mlp", _build_mlp, _mlp_n_params, _suggest_mlp),
}
