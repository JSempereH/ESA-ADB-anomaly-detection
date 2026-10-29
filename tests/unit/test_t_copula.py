import numpy as np

from esa_adb.models.lightweight.t_copula import TCopulaDetector


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    model = TCopulaDetector().fit(X)
    scores = model.score(rng.normal(size=(100, 5)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_detects_broken_cross_channel_correlation():
    rng = np.random.default_rng(1)
    n = 3000
    a = rng.normal(size=n)
    b = 2 * a + rng.normal(scale=0.1, size=n)
    X = np.column_stack([a, b])
    model = TCopulaDetector().fit(X)

    a_test = rng.normal(size=20)
    consistent = np.column_stack([a_test, 2 * a_test + rng.normal(scale=0.1, size=20)])
    broken = np.column_stack([a_test, -2 * a_test + rng.normal(scale=0.1, size=20)])
    assert model.score(broken).mean() > model.score(consistent).mean()


def test_scores_are_nondecreasing_in_mahalanobis_distance():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 3))
    model = TCopulaDetector(df=6.0).fit(X)
    near = rng.normal(scale=0.5, size=(10, 3))
    far = rng.normal(scale=5.0, size=(10, 3))
    assert model.score(far).mean() > model.score(near).mean()


def test_correlation_estimate_matches_gaussian_copula_by_rank_invariance():
    """Spearman rank correlation is invariant to the marginal transform, so the fitted
    correlation matrix should be identical between the Gaussian and t copula."""
    from esa_adb.models.lightweight.gaussian_copula import GaussianCopulaDetector

    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 4))
    g = GaussianCopulaDetector().fit(X)
    t_model = TCopulaDetector().fit(X)
    g_corr = np.linalg.inv(g.corr_inv_)
    t_corr = np.linalg.inv(t_model.corr_inv_)
    assert np.allclose(g_corr, t_corr, atol=1e-9)
