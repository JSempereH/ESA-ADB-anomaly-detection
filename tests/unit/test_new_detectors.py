import numpy as np

from esa_adb.models.lightweight import EchoStateForecaster, HotellingPCA, LodaDetector, VARMahalanobis
from esa_adb.models.runners import CUSUMRunner, FactoryRunner, build_runners, cusum


def _correlated_series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(t / 40.0)
    X = np.stack([base + 0.05 * rng.normal(size=n), 2 * base + 0.05 * rng.normal(size=n), rng.normal(size=n) * 0.5], axis=1)
    return X


def _break_correlation(X, at):
    """Channel 1 is normally 2x channel 0; push it off that line without leaving its own range."""
    X = X.copy()
    X[at, 1] = -X[at, 0] * 2
    return X


def test_var_mahalanobis_flags_correlation_break_that_per_channel_misses():
    X = _correlated_series()
    model = VARMahalanobis().fit(X[:2000])
    probe = _break_correlation(X[2000:2400], 200)
    s = model.score(probe)
    assert s[200] > 5 * np.median(s[10:])


def test_var_mahalanobis_online_matches_shapes_and_zeros_first_lags():
    X = _correlated_series()
    model = VARMahalanobis()
    model.partial_fit(X[:1000]).partial_fit(X[1000:2000])
    s = model.score(X[2000:2100])
    assert s.shape == (100,) and (s[:3] == 0).all() and np.isfinite(s).all()


def test_hotelling_flags_extreme_in_subspace_and_off_subspace():
    X = _correlated_series()
    model = HotellingPCA(n_components=2).fit(X[:2000])
    probe = X[2000:2300].copy()
    probe[100] = probe[100] + np.array([0.0, 0.0, 12.0])  # off-subspace
    probe[200, :2] = probe[200, :2] * 6.0  # on-subspace, extreme
    s = model.score(probe)
    assert s[100] > 3 * np.median(s) and s[200] > 3 * np.median(s)


def test_hotelling_partial_fit_runs():
    X = _correlated_series()
    m = HotellingPCA(n_components=2)
    m.partial_fit(X[:1000]).partial_fit(X[1000:2000])
    assert np.isfinite(m.score(X[2000:2100])).all()


def test_loda_flags_outlier_and_accumulates_online():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 6))
    m = LodaDetector(n_projections=50).fit(X[:2000])
    n1 = m.counts_.sum()
    m.partial_fit(X[2000:3000])
    assert m.counts_.sum() == n1 + 1000 * 50
    probe = X[3000:3100].copy()
    probe[50] = 8.0
    s = m.score(probe)
    assert s[50] > np.percentile(s, 99)


def test_echo_state_flags_a_break_and_is_deterministic():
    X = _correlated_series(2500)
    kw = dict(n_reservoir=60, washout=20)
    a = EchoStateForecaster(**kw).fit(X[:1500])
    b = EchoStateForecaster(**kw).fit(X[:1500])
    probe = _break_correlation(X[1500:1800], 150)
    sa = a.score(probe)
    assert np.allclose(sa, b.score(probe))
    assert sa[150] > 3 * np.median(sa[30:])


def test_cusum_accumulates_persistent_shift_but_not_noise():
    rng = np.random.default_rng(2)
    z = rng.normal(size=500)
    z[300:340] += 1.5  # small persistent shift
    c = cusum(z, k=0.5)
    assert c[339] > 10 * np.median(c[:290]) + 5
    assert c.min() >= 0


def test_cusum_runner_wraps_a_base_runner_and_is_buildable_by_name():
    X = _correlated_series(2000)
    runners = build_runners(["CUSUM-LodaDetector", "VARMahalanobis"], 3, {}, None)
    r = runners["CUSUM-LodaDetector"]
    assert isinstance(r, CUSUMRunner) and isinstance(runners["VARMahalanobis"], FactoryRunner)
    calib, ev = r.online_scores(X[:1200], X[1200:1500], X[1500:1800])
    assert calib.shape == (300,) and ev.shape == (300,) and (ev >= 0).all()
    calib2, ev2 = r.refit_scores(X[:1200], X[1200:1500], X[1500:1800])
    assert ev2.shape == (300,)


def test_chronos_residual_flags_injected_shift():
    import pytest

    pytest.importorskip("chronos")
    from esa_adb.models.lightweight.chronos_residual import ChronosResidualDetector

    rng = np.random.default_rng(0)
    t = np.arange(4000)
    X = np.stack([np.sin(t / 50) + 0.05 * rng.normal(size=4000) + 0.1 * i for i in range(6)], axis=1)
    m = ChronosResidualDetector(fit_blocks=20).fit(X[:2000])
    probe = X[2000:3200].copy()
    probe[640:704, 2] += 3.0
    s = m.score(probe)
    assert (s[:128] == 0).all() and s[640:704].mean() > 2 * np.median(s[128:])


def test_cusum_runner_does_not_ramp_when_fit_scores_are_deflated_in_sample():
    """Regression: an in-sample reference (fit rows score lower than unseen rows) made the
    CUSUM grow without bound into one month-long alarm."""
    from esa_adb.models.runners import ScoreRunner

    class Fake(ScoreRunner):
        name = "Fake"

        def update(self, fit_values):
            pass

        def score_rows(self, values):  # unseen data: noise around 10; in-sample would be ~2
            return 10.0 + np.random.default_rng(len(values)).normal(size=len(values))

    r = CUSUMRunner(Fake())
    r.update(np.zeros((500, 2)))
    calib = r.score_rows(np.zeros((2000, 2)))
    ev = r.score_rows(np.zeros((3000, 2)))
    assert np.median(calib) < 5 and np.median(ev) < 5 and ev.max() < 50
