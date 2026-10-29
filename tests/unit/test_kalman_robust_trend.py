import numpy as np

from esa_adb.models.lightweight.kalman_robust_trend import KalmanRobustTrend


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    model = KalmanRobustTrend().fit(X)
    scores = model.score(rng.normal(size=(100, 4)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_still_flags_anomalies_at_full_magnitude():
    """The score is always the RAW standardized innovation -- clipping only affects
    the state update, never the reported score."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 1))
    model = KalmanRobustTrend(huber_c=1.0).fit(X)
    outlier = np.array([[50.0]])
    assert model.score(outlier)[0] > 10


def test_state_is_less_dragged_by_an_anomaly_than_the_plain_filter():
    """Core hypothesis: after an anomaly, the robust filter's state should stay closer
    to its pre-anomaly value than the plain filter's would."""
    from esa_adb.models.lightweight.kalman_local_trend import KalmanLocalTrend

    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 1))
    robust = KalmanRobustTrend(huber_c=1.0).fit(X)
    plain = KalmanLocalTrend().fit(X)

    state_before_robust = robust.state[0, 0]
    state_before_plain = plain.state[0, 0]
    anomaly = np.array([[100.0]])
    robust.score(anomaly)
    plain.score(anomaly)

    drift_robust = abs(robust.state[0, 0] - state_before_robust)
    drift_plain = abs(plain.state[0, 0] - state_before_plain)
    assert drift_robust < drift_plain


def test_partial_fit_continues_state():
    rng = np.random.default_rng(3)
    model = KalmanRobustTrend().fit(rng.normal(size=(300, 2)))
    state_before = model.state.copy()
    model.partial_fit(rng.normal(size=(100, 2)))
    assert not np.allclose(state_before, model.state)
