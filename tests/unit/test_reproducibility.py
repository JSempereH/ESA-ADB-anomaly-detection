import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.autoencoder import Autoencoder  # noqa: E402
from esa_adb.models.heavy.lstm_forecaster import LSTMForecaster  # noqa: E402
from esa_adb.models.lightweight.pca_reconstruction import PCAReconstructor  # noqa: E402


def _nominal(n=1000, n_channels=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, n_channels))


def test_autoencoder_same_seed_gives_identical_scores():
    X = _nominal()
    a = Autoencoder(n_channels=4, hidden_sizes=(8,), latent_size=2, seed=123)
    a.fit(X, epochs=5, batch_size=64, verbose=False)
    b = Autoencoder(n_channels=4, hidden_sizes=(8,), latent_size=2, seed=123)
    b.fit(X, epochs=5, batch_size=64, verbose=False)

    np.testing.assert_array_equal(a.score(X), b.score(X))


def test_autoencoder_different_seed_gives_different_scores():
    X = _nominal()
    a = Autoencoder(n_channels=4, hidden_sizes=(8,), latent_size=2, seed=1)
    a.fit(X, epochs=5, batch_size=64, verbose=False)
    b = Autoencoder(n_channels=4, hidden_sizes=(8,), latent_size=2, seed=2)
    b.fit(X, epochs=5, batch_size=64, verbose=False)

    assert not np.array_equal(a.score(X), b.score(X))


def test_lstm_forecaster_same_seed_gives_identical_scores():
    X = _nominal(n=300)
    a = LSTMForecaster(n_channels=4, input_length=10, hidden_size=8, num_layers=1, seed=7)
    a.fit(X, epochs=3, batch_size=32, verbose=False)
    b = LSTMForecaster(n_channels=4, input_length=10, hidden_size=8, num_layers=1, seed=7)
    b.fit(X, epochs=3, batch_size=32, verbose=False)

    np.testing.assert_array_equal(a.score(X), b.score(X))


def test_pca_reconstructor_is_already_deterministic_across_fits():
    # No seed parameter needed here -- sklearn's PCA with a variance-ratio n_components
    # uses the exact 'full' solver, not a randomized one, so this should already be
    # bit-identical without any explicit seeding (unlike the two torch models above).
    X = _nominal()
    a = PCAReconstructor(n_components=0.95).fit(X)
    b = PCAReconstructor(n_components=0.95).fit(X)

    np.testing.assert_array_equal(a.score(X), b.score(X))
