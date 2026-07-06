"""Classical baseline models (groundwork §6).

E0 ships only the minimal registry needed to drive the runner end-to-end:
logistic regression and RBF-SVM. The full baseline zoo — SVM-linear, random
forest, XGBoost, k-NN, and the parameter-matched MLP — lands in E1.

Each entry provides:
  build(params, seed) -> unfitted sklearn-compatible estimator
  n_params(estimator) -> number of learned parameters after fit, or None where
                         the notion does not apply (kernel SVMs).
Parameter counts are logged for every run (CLAUDE.md fairness rule); None is
written as an empty CSV cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[dict, int], Any]
    n_params: Callable[[Any], int | None]


def _build_logreg(params: dict, seed: int) -> LogisticRegression:
    defaults: dict[str, Any] = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}
    defaults.update(params)
    return LogisticRegression(random_state=seed, **defaults)


def _logreg_n_params(est: LogisticRegression) -> int:
    return int(est.coef_.size + est.intercept_.size)


def _build_svm_rbf(params: dict, seed: int) -> SVC:
    defaults: dict[str, Any] = {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    defaults.update(params)
    return SVC(random_state=seed, **defaults)


def _svm_n_params(est: SVC) -> None:
    # A support-vector expansion has no fixed parameter count comparable to
    # NN/VQC weights; parameter matching (§8) applies to NN-style models only.
    return None


SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", _build_logreg, _logreg_n_params),
    "svm_rbf": ModelSpec("svm_rbf", _build_svm_rbf, _svm_n_params),
}
