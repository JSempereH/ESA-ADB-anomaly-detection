import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.autoencoder import Autoencoder  # noqa: E402


def test_autoencoder_reconstructs_nominal_correlated_data_well():
    rng = np.random.default_rng(0)
    n = 2000
    latent = rng.normal(0, 1, size=n)
    X = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=n), rng.normal(0, 1, size=n)], axis=1)

    model = Autoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2)
    model.fit(X, epochs=30, batch_size=128, verbose=False)

    assert model.score(X).mean() < 1.0


def test_autoencoder_captures_a_nonlinear_relationship_pca_cannot():
    # channel_2 = channel_1^2 -- zero linear correlation (a linear model/PCA sees no
    # relationship at all here) but a clear, learnable nonlinear one.
    rng = np.random.default_rng(1)
    n = 4000
    c1 = rng.uniform(-1, 1, size=n)
    c2 = c1**2
    train = np.stack([c1, c2], axis=1)

    model = Autoencoder(n_channels=2, hidden_sizes=(16, 8), latent_size=1)
    model.fit(train, epochs=60, batch_size=128, lr=5e-3, verbose=False)
    model.calibrate_threshold(train, percentile=99.0)

    normal_row = np.array([[0.5, 0.25]])  # follows y = x^2
    broken_row = np.array([[0.5, -0.9]])  # same per-channel ranges, relationship violated

    assert model.score(broken_row)[0] > model.score(normal_row)[0] * 2


def test_score_per_channel_shape_and_most_anomalous_channels():
    rng = np.random.default_rng(4)
    n = 2000
    c1 = rng.uniform(-1, 1, size=n)
    c2 = c1**2
    c3 = rng.normal(0, 1, size=n)  # independent, never perturbed below
    train = np.stack([c1, c2, c3], axis=1)

    model = Autoencoder(n_channels=3, hidden_sizes=(16, 8), latent_size=2)
    model.fit(train, epochs=60, batch_size=128, lr=5e-3, verbose=False)

    broken_row = np.array([[0.5, -0.9, 0.1]])  # c1/c2 relationship violated, c3 untouched
    per_channel = model.score_per_channel(broken_row)
    assert per_channel.shape == (1, 3)

    top = model.most_anomalous_channels(broken_row, channel_names=["c1", "c2", "c3"], top_n=1)
    assert top[0][0] in {"c1", "c2"}  # not the untouched independent channel


def test_calibrate_threshold_and_predict():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, size=(1500, 4))
    val = rng.normal(0, 1, size=(400, 4))

    model = Autoencoder(n_channels=4, hidden_sizes=(8,), latent_size=2)
    model.fit(train, epochs=20, batch_size=128, verbose=False)
    model.calibrate_threshold(val, percentile=99.0)

    assert model.threshold_ is not None
    outlier = np.full((1, 4), 50.0)
    assert model.predict(outlier)[0]


def test_predict_without_threshold_raises():
    model = Autoencoder(n_channels=3, hidden_sizes=(4,), latent_size=2)
    model.fit(np.random.default_rng(0).normal(size=(200, 3)), epochs=2, verbose=False)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 3)))


def test_score_before_fit_raises():
    model = Autoencoder(n_channels=3)
    with pytest.raises(RuntimeError):
        model.score(np.zeros((1, 3)))


def test_partial_fit_keeps_first_batch_stats_and_continues_training():
    rng = np.random.default_rng(4)
    batch1 = rng.normal(10, 2, size=(600, 3))  # deliberately non-trivial mean/std
    batch2 = rng.normal(10, 2, size=(600, 3))

    model = Autoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2)
    model.partial_fit(batch1, epochs=5, batch_size=64, verbose=False)
    mean_after_first = model.mean_.copy()

    model.partial_fit(batch2, epochs=5, batch_size=64, verbose=False)
    np.testing.assert_array_equal(model.mean_, mean_after_first)  # never recomputed
    assert model.score(batch2).mean() < model.score(batch1 * 0 + 1000).mean()  # still discriminates


def test_partial_fit_can_be_used_for_calibrate_and_predict_like_fit():
    rng = np.random.default_rng(5)
    train = rng.normal(0, 1, size=(1000, 4))
    val = rng.normal(0, 1, size=(300, 4))

    model = Autoencoder(n_channels=4, hidden_sizes=(8,), latent_size=2)
    model.partial_fit(train, epochs=20, batch_size=128, verbose=False)
    model.calibrate_threshold(val, percentile=99.0)

    outlier = np.full((1, 4), 50.0)
    assert model.predict(outlier)[0]


def test_batched_scoring_matches_unbatched():
    rng = np.random.default_rng(3)
    train = rng.normal(0, 1, size=(500, 3))
    test = rng.normal(0, 1, size=(1000, 3))

    model = Autoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2)
    model.fit(train, epochs=10, batch_size=128, verbose=False)

    full = model.score(test, batch_size=1000)
    chunked = model.score(test, batch_size=137)

    np.testing.assert_allclose(chunked, full, atol=1e-5)
