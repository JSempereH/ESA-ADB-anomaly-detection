import numpy as np
import pytest

from esa_adb.models.lightweight.kalman_local_trend import KalmanLocalTrend


def test_tracks_a_linear_trend_with_low_steady_state_score():
    rng = np.random.default_rng(0)
    n = 2000
    t = np.arange(n)
    X = np.stack([0.01 * t + rng.normal(0, 0.05, n), rng.normal(0, 1, n)], axis=1)
    model = KalmanLocalTrend().fit(X[:500])
    scores = model.score(X[500:1000])
    assert np.median(scores) < 5.0  # tracks the trend, not flooded with false alarms


def test_flags_an_injected_step_change():
    rng = np.random.default_rng(1)
    n = 1000
    X = rng.normal(0, 0.1, size=(n, 2))
    model = KalmanLocalTrend().fit(X[:400])
    probe = X[400:500].copy()
    probe[50, 0] += 20.0  # a sudden, large jump in channel 0
    scores = model.score(probe)
    assert scores[50] > 5 * np.median(scores)


def test_partial_fit_before_fit_falls_back():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    model = KalmanLocalTrend()
    model.partial_fit(X)
    assert model.state is not None


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        KalmanLocalTrend().score(np.zeros((1, 3)))


def test_invalid_obs_noise_fraction_raises():
    with pytest.raises(ValueError):
        KalmanLocalTrend(obs_noise_fraction=1.5)


def test_state_persists_across_sequential_score_calls():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 2))
    model = KalmanLocalTrend().fit(X[:300])
    state_before = model.state.copy()
    model.score(X[300:600])
    assert not np.array_equal(model.state, state_before)  # the filter actually moved
