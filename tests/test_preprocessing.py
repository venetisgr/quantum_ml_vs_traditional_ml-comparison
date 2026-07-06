"""AngleScaler tests — the shared [0, π] feature scaling (groundwork §5/§8)."""
import numpy as np
import pytest

from data.preprocessing import ANGLE_MAX, AngleScaler


def test_train_data_maps_onto_exact_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)) * 10.0 + 5.0
    T = AngleScaler().fit_transform(X)
    assert T.shape == X.shape
    assert T.min() >= 0.0 and T.max() <= ANGLE_MAX
    np.testing.assert_allclose(T.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(T.max(axis=0), ANGLE_MAX, rtol=1e-12)


def test_out_of_range_test_data_is_clipped():
    scaler = AngleScaler().fit(np.array([[0.0], [1.0]]))
    T = scaler.transform(np.array([[-5.0], [0.5], [9.0]]))
    assert T[0, 0] == 0.0
    assert T[2, 0] == ANGLE_MAX
    np.testing.assert_allclose(T[1, 0], ANGLE_MAX / 2)


def test_params_come_from_fit_data_only():
    scaler = AngleScaler().fit(np.array([[0.0], [2.0]]))
    scaler.transform(np.array([[100.0]]))  # must not touch fitted state
    np.testing.assert_array_equal(scaler.min_, [0.0])
    np.testing.assert_array_equal(scaler.range_, [2.0])


def test_constant_feature_maps_to_zero():
    X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    T = AngleScaler().fit_transform(X)
    np.testing.assert_array_equal(T[:, 0], 0.0)
    np.testing.assert_allclose(T[:, 1], [0.0, ANGLE_MAX / 2, ANGLE_MAX])


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        AngleScaler().transform(np.zeros((2, 2)))


def test_fit_rejects_non_2d_and_nonfinite():
    with pytest.raises(ValueError):
        AngleScaler().fit(np.zeros(5))
    with pytest.raises(ValueError):
        AngleScaler().fit(np.array([[np.nan], [1.0]]))
