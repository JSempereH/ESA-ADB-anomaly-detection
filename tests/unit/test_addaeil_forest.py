import numpy as np

from esa_adb.models.lightweight.addaeil_forest import ADDAEILForest


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    model = ADDAEILForest(n_estimators=20).fit(X)
    scores = model.score(rng.normal(size=(100, 4)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_scores_an_obvious_outlier_higher():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    model = ADDAEILForest(n_estimators=20).fit(X)
    inlier = rng.normal(size=(10, 3))
    outlier = np.full((10, 3), 20.0)
    assert model.score(outlier).mean() > model.score(inlier).mean()


def test_partial_fit_replaces_trees_under_strong_drift():
    """A batch with a drastically shifted distribution should trigger tree
    replacement (Module 1's drift intensity should be high, so n_replace > 0)."""
    rng = np.random.default_rng(2)
    model = ADDAEILForest(n_estimators=20, replace_fraction_cap=0.6).fit(rng.normal(size=(1000, 3)))
    trees_before = list(model.model.estimators_)
    model.partial_fit(rng.normal(loc=50, scale=10, size=(1000, 3)))
    trees_after = model.model.estimators_
    assert any(a is not b for a, b in zip(trees_before, trees_after))


def test_partial_fit_replaces_little_under_no_drift():
    rng = np.random.default_rng(3)
    model = ADDAEILForest(n_estimators=20, replace_fraction_cap=0.6).fit(rng.normal(size=(1000, 3)))
    trees_before = list(model.model.estimators_)
    model.partial_fit(rng.normal(size=(1000, 3)))  # same distribution, no real drift
    trees_after = model.model.estimators_
    n_replaced = sum(1 for a, b in zip(trees_before, trees_after) if a is not b)
    assert n_replaced <= 4  # should replace few, if any, trees


def test_replacement_respects_the_fraction_cap():
    rng = np.random.default_rng(4)
    model = ADDAEILForest(n_estimators=20, replace_fraction_cap=0.3).fit(rng.normal(size=(1000, 3)))
    trees_before = list(model.model.estimators_)
    model.partial_fit(rng.normal(loc=100, scale=20, size=(1000, 3)))  # extreme drift
    trees_after = model.model.estimators_
    n_replaced = sum(1 for a, b in zip(trees_before, trees_after) if a is not b)
    assert n_replaced <= 6  # 0.3 * 20 = 6
