import numpy as np
import pytest

from esa_adb.models.lightweight.cluster_regime import ClusterRegimeDetector, _gap_statistic, _mad_filter


def test_mad_filter_flags_injected_outliers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    X[:20] += 20.0  # obvious outlier block
    keep = _mad_filter(X, k=3.5, max_reject_fraction=0.1)
    assert keep[:20].sum() < 20  # most/all injected outliers get rejected
    assert keep[20:].mean() > 0.9  # the rest stay mostly intact


def test_gap_statistic_prefers_two_clusters_for_two_well_separated_blobs():
    rng = np.random.default_rng(1)
    blob_a = rng.normal(0, 0.3, size=(100, 2))
    blob_b = rng.normal(10, 0.3, size=(100, 2))
    X = np.concatenate([blob_a, blob_b])
    k = _gap_statistic(X, range(1, 6), b=5, seed=1)
    assert k == 2


def test_fit_and_score_finite():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 5))
    model = ClusterRegimeDetector(k_range=range(2, 5), gap_bootstrap=3, lof_neighbors=10)
    model.fit(X)
    scores = model.score(rng.normal(size=(200, 5)))
    assert scores.shape == (200,)
    assert np.isfinite(scores).all()


def test_scores_an_obvious_outlier_higher():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 4))
    model = ClusterRegimeDetector(k_range=range(2, 5), gap_bootstrap=3, lof_neighbors=10).fit(X)
    inlier = rng.normal(size=(5, 4))
    outlier = np.full((5, 4), 15.0)
    assert model.score(outlier).mean() > model.score(inlier).mean()


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        ClusterRegimeDetector().score(np.zeros((1, 3)))
