"""Model-zoo contract tests: build, fit, predict, n_params, search spaces."""
import numpy as np
import optuna
import pytest

from data import load_dataset
from data.preprocessing import AngleScaler
from models import SPECS, available_models, get_spec

E1_ZOO = ["logreg", "svm_linear", "svm_rbf", "knn", "rf", "xgboost", "mlp"]

# Fixed n_params expectations on 2 features (moons); None = not applicable.
N_PARAMS_2D = {
    "logreg": 3,          # 2 coefficients + intercept
    "svm_linear": 3,
    "svm_rbf": None,
    "knn": None,
    "rf": None,
    "xgboost": None,
    "mlp": 8 * 2 + 8 + 8 + 1,   # default width 8: (2->8->1) weights + biases
}


@pytest.fixture(scope="module")
def moons_fold():
    X, y, _ = load_dataset("moons")
    scaler = AngleScaler().fit(X[:400])
    return scaler.transform(X[:400]), y[:400], scaler.transform(X[400:]), y[400:]


def test_registry_is_the_e1_zoo():
    assert available_models() == sorted(E1_ZOO)


@pytest.mark.parametrize("name", E1_ZOO)
def test_build_fit_predict_and_n_params(name, moons_fold):
    X_train, y_train, X_test, y_test = moons_fold
    spec = get_spec(name)
    model = spec.build({}, seed=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == y_test.shape
    assert (pred == y_test).mean() > 0.75, f"{name} unreasonably bad on moons"
    assert spec.n_params(model) == N_PARAMS_2D[name]
    # every model must expose scores for ROC-AUC
    assert hasattr(model, "predict_proba") or hasattr(model, "decision_function")


@pytest.mark.parametrize("name", E1_ZOO)
def test_search_space_produces_valid_buildable_params(name):
    spec = get_spec(name)
    assert spec.suggest is not None, f"{name} has no search space"
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(3):
        trial = study.ask()
        params = spec.suggest(trial)
        assert isinstance(params, dict) and params
        assert len(params) <= 6, "search-space size philosophy: <= 6 hyperparameters"
        model = spec.build(dict(params), seed=0)  # must construct
        assert model is not None
        study.tell(trial, 0.5)


def test_mlp_search_space_bounds():
    spec = get_spec("mlp")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    for _ in range(10):
        trial = study.ask()
        params = spec.suggest(trial)
        (width,) = params["hidden_layer_sizes"]
        assert 2 <= width <= 32
        assert 1e-6 <= params["alpha"] <= 1e-1
        study.tell(trial, 0.5)


def test_mlp_n_params_formula():
    X, y, _ = load_dataset("iris_binary")  # 4 features
    spec = get_spec("mlp")
    model = spec.build({"hidden_layer_sizes": [3], "max_iter": 50}, seed=0)
    model.fit(X, y)
    assert spec.n_params(model) == 4 * 3 + 3 + 3 + 1  # 19


def test_seed_reaches_stochastic_models(moons_fold):
    X_train, y_train, X_test, _ = moons_fold
    spec = get_spec("rf")
    preds = []
    for seed in (0, 0, 1):
        model = spec.build({"n_estimators": 30}, seed=seed)
        model.fit(X_train, y_train)
        preds.append(model.predict_proba(X_test)[:, 1])
    np.testing.assert_array_equal(preds[0], preds[1])  # same seed -> identical
    assert not np.array_equal(preds[0], preds[2])      # different seed -> differs
