import numpy as np

from esa_adb.models.lightweight.kalman_adaptive_trend import KalmanAdaptiveTrend


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    model = KalmanAdaptiveTrend().fit(X)
    scores = model.score(rng.normal(size=(100, 4)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_r_adapts_upward_when_noise_floor_genuinely_widens():
    rng = np.random.default_rng(1)
    X = rng.normal(scale=1.0, size=(2000, 1))
    model = KalmanAdaptiveTrend(adapt_rate=0.05, gate=3.0).fit(X)
    r_before = model.R[0]
    wider_noise = rng.normal(scale=5.0, size=(2000, 1))
    model.score(wider_noise)
    assert model.R[0] > r_before


def test_gating_blocks_adaptation_from_a_single_large_anomaly():
    """An anomalous row (beyond `gate`) must not be allowed to inflate R."""
    rng = np.random.default_rng(2)
    X = rng.normal(scale=1.0, size=(2000, 1))
    model = KalmanAdaptiveTrend(adapt_rate=0.05, gate=3.0).fit(X)
    r_before = model.R[0]
    anomaly = np.array([[1000.0]])
    model.score(anomaly)
    assert model.R[0] == r_before


def test_partial_fit_continues_state():
    rng = np.random.default_rng(3)
    model = KalmanAdaptiveTrend().fit(rng.normal(size=(300, 2)))
    r_before = model.R.copy()
    model.partial_fit(rng.normal(scale=3.0, size=(300, 2)))
    assert not np.allclose(r_before, model.R)
