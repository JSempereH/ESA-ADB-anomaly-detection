import numpy as np

from esa_adb.models.lightweight.gbm_forecaster import GBMForecaster


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    model = GBMForecaster(n_lags=2, n_estimators=10, max_buffer_rows=1000).fit(X)
    scores = model.score(rng.normal(size=(100, 4)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_first_n_lags_rows_score_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    model = GBMForecaster(n_lags=3, n_estimators=10, max_buffer_rows=1000).fit(X)
    scores = model.score(rng.normal(size=(50, 3)))
    assert (scores[:3] == 0).all()


def test_scores_an_obvious_outlier_higher():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 3))
    model = GBMForecaster(n_lags=2, n_estimators=20, max_buffer_rows=2000).fit(X)
    inlier = rng.normal(size=(10, 3))
    outlier = np.full((10, 3), 20.0)
    assert model.score(outlier).mean() > model.score(inlier).mean()


def test_partial_fit_respects_buffer_cap():
    rng = np.random.default_rng(3)
    model = GBMForecaster(n_lags=2, n_estimators=5, max_buffer_rows=300)
    model.fit(rng.normal(size=(200, 2)))
    model.partial_fit(rng.normal(size=(200, 2)))
    assert model._buffer.shape[0] == 300


def test_score_per_channel_shape():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 5))
    model = GBMForecaster(n_lags=2, n_estimators=10, max_buffer_rows=500).fit(X)
    z = model.score_per_channel(rng.normal(size=(50, 5)))
    assert z.shape == (50, 5)
    assert np.isfinite(z).all()
