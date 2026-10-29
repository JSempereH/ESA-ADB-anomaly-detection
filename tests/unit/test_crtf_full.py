import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.crtf_full import CRTFFull, _finch_clusters  # noqa: E402


def test_finch_clusters_separate_two_distant_blobs():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(0, 0.1, size=(20, 4))
    blob_b = rng.normal(10, 0.1, size=(20, 4))
    z = np.concatenate([blob_a, blob_b])
    labels = _finch_clusters(z)
    assert len(set(labels[:20])) < len(set(labels))  # at least some within-blob merging happened
    assert labels[0] != labels[-1]  # the two far-apart blobs never end up in the same cluster


def test_fit_and_score_finite():
    rng = np.random.default_rng(1)
    latent = rng.normal(0, 1, size=1500)
    X = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=1500), rng.normal(0, 1, size=1500)], axis=1)
    model = CRTFFull(n_channels=3, window_length=10, hidden_time=4, hidden_feat=4, n_layers=1, k_neighbors=5, min_cluster_size=3)
    model.fit(X, epochs=3, batch_size=64, stride=2, verbose=False)
    scores = model.score(X[:300])
    assert scores.shape == (300,)
    assert np.isfinite(scores).all()
    # Unlike a forecaster, CRTF reconstructs the window itself (non-causal), so even
    # row 0 is covered by the window starting there -- no "no context yet" gap.
    assert (scores > 0).any()


def test_partial_fit_before_fit_falls_back_and_keeps_stats():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 3))
    model = CRTFFull(n_channels=3, window_length=10, hidden_time=4, hidden_feat=4, n_layers=1, min_cluster_size=3)
    model.partial_fit(X, epochs=2, batch_size=64, stride=2, verbose=False)
    mean_after_first = model.mean_.copy()
    model.partial_fit(rng.normal(size=(400, 3)), epochs=2, batch_size=64, stride=2, verbose=False)
    np.testing.assert_array_equal(model.mean_, mean_after_first)


def test_score_before_fit_raises():
    model = CRTFFull(n_channels=3, window_length=10)
    with pytest.raises(RuntimeError):
        model.score(np.zeros((50, 3)))


def test_score_too_short_input_returns_zeros():
    rng = np.random.default_rng(3)
    model = CRTFFull(n_channels=3, window_length=20, hidden_time=4, hidden_feat=4, n_layers=1, min_cluster_size=3)
    model.fit(rng.normal(size=(600, 3)), epochs=2, batch_size=64, stride=2, verbose=False)
    short = rng.normal(size=(5, 3))
    np.testing.assert_array_equal(model.score(short), np.zeros(5))
