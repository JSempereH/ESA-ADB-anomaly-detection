import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.roca_full import RoCAFull  # noqa: E402


def _model(**kwargs):
    defaults = dict(n_channels=3, window_length=24, tcn_hidden=8, lstm_hidden=8, proj_size=8, nu=0.07)
    defaults.update(kwargs)
    return RoCAFull(**defaults)


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 3))
    model = _model()
    model.fit(X, epochs=3, batch_size=64, stride=2, verbose=False)
    scores = model.score(X[:300])
    assert scores.shape == (300,)
    assert np.isfinite(scores).all()


def test_center_fixed_after_first_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 3))
    model = _model()
    model.fit(X, epochs=2, batch_size=64, stride=2, verbose=False)
    center = model.center.clone()
    model.partial_fit(rng.normal(size=(500, 3)), epochs=2, batch_size=64, stride=2, verbose=False)
    assert torch.allclose(model.center, center)


def test_score_too_short_input_returns_zeros():
    rng = np.random.default_rng(2)
    model = _model()
    model.fit(rng.normal(size=(600, 3)), epochs=2, batch_size=64, stride=2, verbose=False)
    short = rng.normal(size=(5, 3))
    np.testing.assert_array_equal(model.score(short), np.zeros(5))


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().score(np.zeros((50, 3)))


def test_window_length_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        _model(window_length=10)


def test_partial_fit_before_fit_falls_back():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 3))
    model = _model()
    model.partial_fit(X, epochs=2, batch_size=64, stride=2, verbose=False)
    assert model.mean_ is not None and model.center is not None
