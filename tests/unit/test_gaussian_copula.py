import numpy as np

from esa_adb.models.lightweight.gaussian_copula import GaussianCopulaDetector


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    model = GaussianCopulaDetector().fit(X)
    scores = model.score(rng.normal(size=(100, 5)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_robust_to_contaminated_fit_batch():
    """A handful of extreme rows in the (unfiltered, L2-style) fit batch should not
    blow up the fitted correlation structure -- the same contamination-robustness bug
    the Kalman block extension needed a fix for."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 4))
    X[:50] += rng.normal(scale=1000, size=(50, 4))  # contamination
    model = GaussianCopulaDetector().fit(X)
    normal_eval = rng.normal(size=(200, 4))
    scores = model.score(normal_eval)
    assert np.isfinite(scores).all()
    # Normal-looking eval rows shouldn't register as extreme outliers just because the
    # fit batch had contamination.
    assert np.median(scores) < 5.0


def test_detects_broken_cross_channel_correlation():
    """Same hypothesis as the Kalman block extension: two correlated channels behaving
    individually-unremarkably but breaking their normal relationship should score
    higher than a correlation-consistent row."""
    rng = np.random.default_rng(2)
    n = 3000
    a = rng.normal(size=n)
    b = 2 * a + rng.normal(scale=0.1, size=n)
    X = np.column_stack([a, b])
    model = GaussianCopulaDetector().fit(X)

    a_test = rng.normal(size=20)
    consistent = np.column_stack([a_test, 2 * a_test + rng.normal(scale=0.1, size=20)])
    broken = np.column_stack([a_test, -2 * a_test + rng.normal(scale=0.1, size=20)])

    assert model.score(broken).mean() > model.score(consistent).mean()


def test_pit_handles_values_outside_the_reference_range():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 3))
    model = GaussianCopulaDetector().fit(X)
    extreme = np.full((5, 3), 1000.0)
    scores = model.score(extreme)
    assert np.isfinite(scores).all()


def test_partial_fit_refits_from_the_new_batch():
    rng = np.random.default_rng(4)
    model = GaussianCopulaDetector().fit(rng.normal(size=(1000, 3)))
    ref_before = model.reference_.copy()
    model.partial_fit(rng.normal(loc=50, size=(1000, 3)))
    assert not np.allclose(ref_before, model.reference_)
