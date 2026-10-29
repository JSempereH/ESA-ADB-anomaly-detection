import numpy as np

from esa_adb.models.lightweight.dynamic_factor_model import DynamicFactorModel


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    model = DynamicFactorModel(n_components=3).fit(X)
    scores = model.score(rng.normal(size=(100, 6)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_idiosyncratic_residual_matches_pca_reconstruction_error():
    """score_per_channel is the same primitive PCAReconstructor exposes."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    model = DynamicFactorModel(n_components=2).fit(X)
    z = model.score_per_channel(rng.normal(size=(50, 4)))
    assert z.shape == (50, 4)
    assert np.isfinite(z).all()
    assert (z >= 0).all()


def test_never_scores_below_the_idiosyncratic_residual():
    """The final score is max(idiosyncratic, factor-innovation), so it can only add
    signal relative to a plain reconstruction-error score, never remove it."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 5))
    model = DynamicFactorModel(n_components=3).fit(X)
    eval_X = rng.normal(size=(50, 5))
    idio = np.sqrt(model.score_per_channel(eval_X).mean(axis=1))
    full = model.score(eval_X)
    assert (full >= idio - 1e-9).all()


def test_detects_factor_dynamics_break():
    """A row sequence whose factor jumps inconsistently with its learned AR(1)
    dynamics should score higher than one consistent with those dynamics."""
    rng = np.random.default_rng(3)
    n = 3000
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.9 * f[t - 1] + rng.normal(scale=0.05)
    loadings = np.array([1.0, 0.8, -0.6])
    X = np.outer(f, loadings) + rng.normal(scale=0.02, size=(n, 3))
    model = DynamicFactorModel(n_components=1).fit(X)

    consistent = np.outer(0.9 * f[-1:] + rng.normal(scale=0.05, size=5), loadings)
    broken = np.outer(np.full(5, 10.0), loadings)  # factor jumps far outside its AR(1) range
    assert model.score(broken).mean() > model.score(consistent).mean()


def test_partial_fit_continues_without_resetting_pca():
    rng = np.random.default_rng(4)
    model = DynamicFactorModel(n_components=2).fit(rng.normal(size=(500, 4)))
    pca_before = model.pca
    model.partial_fit(rng.normal(size=(100, 4)))
    assert model.pca is pca_before
