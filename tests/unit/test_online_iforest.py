import numpy as np
import pytest

from esa_adb.models.lightweight.online_iforest import OnlineIsolationForest


def test_scores_outliers_higher_than_inliers():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(4000, 3))
    model = OnlineIsolationForest(n_trees=16, omega=512, eta=16, stride=1).fit(X)

    inlier = np.zeros((1, 3))
    outlier = np.full((1, 3), 9.0)
    assert model.score(outlier)[0] > model.score(inlier)[0]


def test_partial_fit_adapts_to_a_shifted_regime():
    rng = np.random.default_rng(1)
    old = rng.normal(0, 1, size=(3000, 2))
    new = rng.normal(6, 1, size=(3000, 2))
    model = OnlineIsolationForest(n_trees=16, omega=512, eta=16, stride=1).fit(old)

    probe_new = rng.normal(6, 1, size=(200, 2))
    before = model.score(probe_new).mean()
    model.partial_fit(new)
    after = model.score(probe_new).mean()

    assert after < before  # higher score = more anomalous; the new regime stops looking anomalous


def test_forgetting_bounds_the_window():
    rng = np.random.default_rng(2)
    model = OnlineIsolationForest(n_trees=4, omega=64, eta=8, stride=1).fit(rng.normal(size=(100, 2)))
    model.partial_fit(rng.normal(size=(500, 2)))
    assert len(model._window) == 64


def test_same_seed_is_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1000, 4))
    probe = rng.normal(size=(100, 4))
    a = OnlineIsolationForest(n_trees=8, omega=128, eta=8, stride=2, seed=7).fit(X).score(probe)
    b = OnlineIsolationForest(n_trees=8, omega=128, eta=8, stride=2, seed=7).fit(X).score(probe)
    np.testing.assert_array_equal(a, b)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        OnlineIsolationForest().score(np.zeros((1, 2)))


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        OnlineIsolationForest(omega=8, eta=16)
