import numpy as np

from esa_adb.models.lightweight import LaggedRidgeForecaster


def _ar_data(n_rows: int = 2000, n_channels: int = 4, seed: int = 0) -> np.ndarray:
    """Channels with strong AR(1) structure a lag-forecaster must be able to learn."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_rows, n_channels))
    X[0] = rng.normal(size=n_channels)
    for t in range(1, n_rows):
        X[t] = 0.95 * X[t - 1] + 0.05 * rng.normal(size=n_channels)
    return X


class TestLaggedRidgeForecaster:
    def test_spike_scores_far_above_nominal(self):
        X = _ar_data()
        model = LaggedRidgeForecaster(n_lags=3).fit(X)
        X_test = _ar_data(seed=1)
        X_anom = X_test.copy()
        X_anom[1000, 2] += 20.0
        scores = model.score(X_anom)
        nominal_level = np.percentile(model.score(X_test), 99)
        assert scores[1000] > 5 * nominal_level

    def test_leading_rows_score_zero(self):
        X = _ar_data()
        model = LaggedRidgeForecaster(n_lags=3).fit(X)
        assert (model.score(X)[:3] == 0).all()

    def test_partial_fit_equals_fit_on_same_data(self):
        # Accumulating sufficient statistics in two chunks must give the same solution
        # as one batch fit over the concatenation of those chunks (up to the boundary
        # rows lost between chunks and the residual-std blending).
        X = _ar_data()
        batch = LaggedRidgeForecaster(n_lags=2).fit(X)
        online = LaggedRidgeForecaster(n_lags=2)
        online.partial_fit(X[:1000])
        online.partial_fit(X[1000:])
        np.testing.assert_allclose(online.weights_, batch.weights_, atol=1e-3)

    def test_partial_fit_adapts_to_new_regime(self):
        X = _ar_data()
        model = LaggedRidgeForecaster(n_lags=2).fit(X)
        shifted = _ar_data(seed=2) + 10.0
        before = model.score(shifted).mean()
        for _ in range(5):
            model.partial_fit(shifted)
        after = model.score(shifted).mean()
        assert after < before

    def test_score_per_channel_identifies_offender(self):
        X = _ar_data()
        model = LaggedRidgeForecaster(n_lags=3).fit(X)
        X_anom = _ar_data(seed=1)
        X_anom[500, 1] += 20.0
        assert model.score_per_channel(X_anom)[500].argmax() == 1
        # local index 5 in the slice = global row 500; the slice must include at least
        # n_lags rows of context before it, or the row falls in the zero-scored lead-in.
        assert model.most_anomalous_channels(X_anom[495:502], top_n=1)[5] == [1]
