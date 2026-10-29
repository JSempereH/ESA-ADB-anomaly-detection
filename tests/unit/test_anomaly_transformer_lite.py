import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.anomaly_transformer_lite import AnomalyTransformerLite  # noqa: E402


def _model(**kwargs):
    defaults = dict(n_channels=3, window_length=30, d_model=16, n_heads=4, n_layers=2, ff_hidden=32)
    defaults.update(kwargs)
    return AnomalyTransformerLite(**defaults)


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 3))
    model = _model()
    model.fit(X, epochs=3, batch_size=64, stride=2, verbose=False)
    scores = model.score(X[:300])
    assert scores.shape == (300,)
    assert np.isfinite(scores).all()


def test_detects_and_localizes_an_injected_spike():
    rng = np.random.default_rng(7)
    latent = rng.normal(0, 1, size=3000)
    X = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=3000), rng.normal(0, 1, size=3000)], axis=1)
    model = _model(seed=7)
    model.fit(X, epochs=15, batch_size=128, stride=2, verbose=False)

    probe = X[500:560].copy()
    spiked = probe.copy()
    spiked[30, :] = 15.0

    s_normal = model.score(probe)
    s_spiked = model.score(spiked)
    assert s_spiked.max() > 10 * s_normal.max()
    assert s_spiked.argmax() == 30


def test_partial_fit_before_fit_falls_back():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    model = _model()
    model.partial_fit(X, epochs=2, batch_size=64, stride=2, verbose=False)
    assert model.mean_ is not None


def test_score_too_short_input_returns_zeros():
    rng = np.random.default_rng(2)
    model = _model()
    model.fit(rng.normal(size=(600, 3)), epochs=2, batch_size=64, stride=2, verbose=False)
    short = rng.normal(size=(5, 3))
    np.testing.assert_array_equal(model.score(short), np.zeros(5))


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().score(np.zeros((50, 3)))


def test_d_model_must_be_divisible_by_n_heads():
    with pytest.raises(ValueError):
        _model(d_model=15, n_heads=4)
