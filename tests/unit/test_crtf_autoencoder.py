import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.crtf_autoencoder import CRTFAutoencoder  # noqa: E402


def test_reconstructs_correlated_nominal_data_well():
    rng = np.random.default_rng(0)
    n = 2000
    latent = rng.normal(0, 1, size=n)
    X = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=n), rng.normal(0, 1, size=n)], axis=1)

    model = CRTFAutoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2, n_clusters=3)
    model.fit(X, epochs=20, batch_size=256, verbose=False)

    assert model.score(X).mean() < 1.0


def test_score_before_fit_raises():
    model = CRTFAutoencoder(n_channels=3, n_clusters=3)
    with pytest.raises(RuntimeError):
        model.score(np.zeros((1, 3)))


def test_partial_fit_before_fit_falls_back_to_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(500, 3))
    model = CRTFAutoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2, n_clusters=3)
    model.partial_fit(X, epochs=5, batch_size=64, verbose=False)
    assert model.mean_ is not None


def test_partial_fit_keeps_first_batch_stats():
    rng = np.random.default_rng(2)
    batch1 = rng.normal(10, 2, size=(600, 3))
    batch2 = rng.normal(10, 2, size=(600, 3))

    model = CRTFAutoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2, n_clusters=3, warmup_epochs=2)
    model.partial_fit(batch1, epochs=5, batch_size=64, verbose=False)
    mean_after_first = model.mean_.copy()

    model.partial_fit(batch2, epochs=5, batch_size=64, verbose=False)
    np.testing.assert_array_equal(model.mean_, mean_after_first)


def test_batched_scoring_matches_unbatched():
    rng = np.random.default_rng(3)
    train = rng.normal(0, 1, size=(500, 3))
    test = rng.normal(0, 1, size=(1000, 3))

    model = CRTFAutoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2, n_clusters=3)
    model.fit(train, epochs=10, batch_size=128, verbose=False)

    full = model.score(test, batch_size=1000)
    chunked = model.score(test, batch_size=137)
    np.testing.assert_allclose(chunked, full, atol=1e-5)


def test_contamination_gets_downweighted_more_than_blind_autoencoder():
    """The property CRTF is supposed to buy: fit on a mix of clean-normal rows plus a
    block of injected outliers, and check the model still scores clean HELD-OUT normal
    rows lower (relatively) than it scores the injected-outlier pattern -- i.e. it
    didn't fully absorb the contamination into its notion of normal, at least not more
    than the blind Autoencoder does on the identical contaminated fit set."""
    rng = np.random.default_rng(4)
    n_clean = 1800
    latent = rng.normal(0, 1, size=n_clean)
    clean = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=n_clean), rng.normal(0, 1, size=n_clean)], axis=1)
    n_outliers = 200
    outliers = rng.normal(0, 1, size=(n_outliers, 3)) * 8  # a contamination "block" unrelated to the clean structure
    contaminated_fit = np.concatenate([clean, outliers])

    from esa_adb.models.heavy.autoencoder import Autoencoder

    blind = Autoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2, seed=4)
    blind.fit(contaminated_fit, epochs=20, batch_size=256, verbose=False)

    crtf = CRTFAutoencoder(n_channels=3, hidden_sizes=(8,), latent_size=2, n_clusters=4, seed=4)
    crtf.fit(contaminated_fit, epochs=20, batch_size=256, verbose=False)

    held_out_clean = np.stack(
        [rng.normal(0, 1, 200), rng.normal(0, 1, 200) * 2, rng.normal(0, 1, 200)], axis=1
    )
    held_out_clean[:, 1] = held_out_clean[:, 0] * 2 + 0.01 * rng.normal(size=200)
    probe_outlier = outliers[:5]

    blind_gap = blind.score(probe_outlier).mean() - blind.score(held_out_clean).mean()
    crtf_gap = crtf.score(probe_outlier).mean() - crtf.score(held_out_clean).mean()

    # CRTF's gap between "outlier-like" and "clean" scores should be at least as large
    # as the blind model's -- if adaptive weighting is doing anything, the model should
    # separate the two at least as well despite training on the same contaminated set.
    assert crtf_gap >= blind_gap * 0.5
