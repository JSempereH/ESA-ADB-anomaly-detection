import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.memstream import MemStream  # noqa: E402


def test_memstream_scores_correlated_nominal_data_low():
    rng = np.random.default_rng(0)
    n = 2000
    latent = rng.normal(0, 1, size=n)
    X = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=n), rng.normal(0, 1, size=n)], axis=1)

    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=256)
    model.fit(X, epochs=30, batch_size=128, verbose=False)

    normal_row = X[:5]
    broken_row = np.array([[5.0, -5.0, 5.0]])
    assert model.score(broken_row)[0] > model.score(normal_row).mean()


def test_memory_seeded_at_fit_time():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(500, 3))
    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=128)
    model.fit(X, epochs=5, batch_size=64, verbose=False)
    assert model.memory_ is not None
    assert model.memory_.shape == (128, 2)
    assert model.beta_ is not None


def test_partial_fit_never_touches_encoder_weights():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(500, 3))
    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=128)
    model.fit(X, epochs=10, batch_size=64, verbose=False)
    weights_before = [p.detach().clone() for p in model.net.parameters()]

    model.partial_fit(rng.normal(0, 1, size=(300, 3)), verbose=False)

    for before, after in zip(weights_before, model.net.parameters()):
        np.testing.assert_array_equal(before.numpy(), after.detach().numpy())


def test_partial_fit_never_admits_outliers_into_memory():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, size=(1000, 3))
    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=64)
    model.fit(X, epochs=20, batch_size=64, verbose=False)
    memory_before = model.memory_.copy()

    outliers = np.full((500, 3), 100.0)
    model.partial_fit(outliers, verbose=False)

    np.testing.assert_array_equal(model.memory_, memory_before)


def test_partial_fit_before_fit_falls_back_to_fit():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, size=(300, 3))
    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=64)
    model.partial_fit(X, verbose=False)
    assert model.memory_ is not None


def test_score_before_fit_raises():
    model = MemStream(n_channels=3, mem_size=64)
    with pytest.raises(RuntimeError):
        model.score(np.zeros((1, 3)))


def test_batched_scoring_matches_unbatched():
    rng = np.random.default_rng(5)
    train = rng.normal(0, 1, size=(500, 3))
    test = rng.normal(0, 1, size=(1000, 3))

    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=128)
    model.fit(train, epochs=10, batch_size=128, verbose=False)

    full = model.score(test, batch_size=1000)
    chunked = model.score(test, batch_size=137)
    np.testing.assert_allclose(chunked, full, atol=1e-5)


def test_mem_size_larger_than_fit_batch_pads_without_error():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, size=(50, 3))
    model = MemStream(n_channels=3, hidden_sizes=(8,), latent_size=2, mem_size=200)
    model.fit(X, epochs=5, batch_size=32, verbose=False)
    assert model.memory_.shape == (200, 2)
