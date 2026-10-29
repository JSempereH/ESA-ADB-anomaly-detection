import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.plda_autoencoder import PLDAAutoencoder, plda_weights  # noqa: E402
from esa_adb.models.heavy.roca_lite import RoCALite  # noqa: E402


def test_plda_weights_penalize_contamination_and_boost_hard_normals():
    # rows 0..4: high loss, high behavior ("anomaly contamination")
    # rows 5..9: high loss, low behavior ("hard normal")
    # rows 10..14: low loss (simple normal)
    loss = np.concatenate([np.linspace(0.9, 1.0, 5), np.linspace(0.8, 0.85, 5), np.linspace(0.0, 0.1, 5)])
    behavior = np.concatenate([np.linspace(0.9, 1.0, 5), np.linspace(0.0, 0.1, 5), np.linspace(0.4, 0.5, 5)])
    w = plda_weights(loss, behavior)

    assert w[:5].mean() < w[10:].mean() < w[5:10].mean()
    assert w.min() >= 0.05 and w.max() <= 2.0


def test_plda_fit_and_partial_fit_keep_mean_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(800, 3))
    model = PLDAAutoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2)
    model.fit(X, epochs=6, batch_size=128, verbose=False)
    mean_before = model.mean_.copy()
    model.partial_fit(rng.normal(0, 1, size=(400, 3)), epochs=3, batch_size=128, verbose=False)
    np.testing.assert_array_equal(model.mean_, mean_before)
    assert np.isfinite(model.score(X)).all()


def test_roca_center_is_fixed_after_first_fit_and_labels_match_nu():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(1000, 3))
    model = RoCALite(n_channels=3, nu=0.07, hidden_sizes=(8,), latent_size=2)
    model.fit(X, epochs=3, batch_size=256, verbose=False)
    center = model.center.clone()
    model.partial_fit(rng.normal(0, 1, size=(500, 3)), epochs=2, batch_size=256, verbose=False)
    assert torch.allclose(model.center, center)


def test_roca_labels_top_nu_fraction(capsys):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(1000, 3))
    model = RoCALite(n_channels=3, nu=0.07, hidden_sizes=(8,), latent_size=2)
    model.fit(X, epochs=2, batch_size=256, verbose=True)
    out = capsys.readouterr().out
    assert "labeled_anomalies=70" in out  # ceil(0.07 * 1000)


def test_roca_score_before_fit_raises_and_scores_finite():
    with pytest.raises(RuntimeError):
        RoCALite(n_channels=3).score(np.zeros((1, 3)))
    rng = np.random.default_rng(3)
    model = RoCALite(n_channels=3, nu=0.07, hidden_sizes=(8,), latent_size=2)
    model.fit(rng.normal(size=(500, 3)), epochs=2, batch_size=128, verbose=False)
    assert np.isfinite(model.score(rng.normal(size=(50, 3)))).all()
