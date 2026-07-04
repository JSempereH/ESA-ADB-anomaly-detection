import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA

from esa_adb.models.lightweight.pca_reconstruction import PCAReconstructor


def test_pca_reconstructs_nominal_correlated_data_well():
    rng = np.random.default_rng(0)
    n = 3000
    latent = rng.normal(0, 1, size=n)
    # Two channels perfectly correlated (both driven by the same latent factor) + a
    # third, independent channel -- a 1-component PCA should capture the first two well.
    X = np.stack([latent, latent * 2 + 0.01 * rng.normal(size=n), rng.normal(0, 1, size=n)], axis=1)

    model = PCAReconstructor(n_components=2).fit(X)
    scores = model.score(X)
    assert scores.mean() < 1.0  # nominal reconstruction error should be small


def test_pca_flags_correlation_break_that_keeps_each_channel_in_range():
    # The exact failure mode GlobalSTD can't see: each channel individually stays within
    # its own nominal range, but the relationship between them breaks.
    rng = np.random.default_rng(1)
    n = 3000
    latent = rng.normal(0, 1, size=n)
    train = np.stack([latent, latent, rng.normal(0, 1, size=n)], axis=1)  # channel_1 == channel_2

    model = PCAReconstructor(n_components=1).fit(train)

    normal_row = np.array([[0.5, 0.5, 0.1]])  # follows the learned correlation
    broken_correlation_row = np.array([[0.5, -0.5, 0.1]])  # same per-channel range, correlation violated

    assert model.score(broken_correlation_row)[0] > model.score(normal_row)[0] * 2


def test_pca_calibrate_threshold_and_predict():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, size=(2000, 4))
    val = rng.normal(0, 1, size=(500, 4))

    model = PCAReconstructor(n_components=0.95).fit(train)
    model.calibrate_threshold(val, percentile=99.0)

    assert model.threshold_ is not None
    outlier = np.full((1, 4), 50.0)
    assert model.predict(outlier)[0]
    # A typical validation point (the calibration data itself) shouldn't trip a
    # 99th-percentile-of-that-same-data threshold.
    assert not model.predict(val[:1])[0]


def test_pca_predict_without_threshold_raises():
    model = PCAReconstructor().fit(np.random.default_rng(0).normal(size=(100, 3)))
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 3)))


def test_partial_fit_switches_to_incremental_pca_and_keeps_first_batch_stats():
    rng = np.random.default_rng(3)
    batch1 = rng.normal(10, 2, size=(500, 4))  # deliberately non-trivial mean/std
    batch2 = rng.normal(10, 2, size=(500, 4))

    model = PCAReconstructor(n_components=0.95)
    model.partial_fit(batch1)
    mean_after_first = model.mean_.copy()
    assert isinstance(model.pca, IncrementalPCA)

    model.partial_fit(batch2)
    np.testing.assert_array_equal(model.mean_, mean_after_first)  # never recomputed


def test_partial_fit_learns_correlation_structure_incrementally():
    rng = np.random.default_rng(4)
    n = 4000
    latent = rng.normal(0, 1, size=n)
    X = np.stack([latent, latent, rng.normal(0, 1, size=n)], axis=1)  # channel_1 == channel_2

    model = PCAReconstructor(n_components=1)
    for start in range(0, n, 400):  # stream in chunks, like successive walk-forward folds
        model.partial_fit(X[start : start + 400])

    normal_row = np.array([[0.5, 0.5, 0.1]])
    broken_correlation_row = np.array([[0.5, -0.5, 0.1]])
    assert model.score(broken_correlation_row)[0] > model.score(normal_row)[0] * 2


def test_score_per_channel_and_most_anomalous_channels_identify_the_broken_pair():
    rng = np.random.default_rng(20)
    n = 3000
    latent = rng.normal(0, 1, size=n)
    train = np.stack([latent, latent, rng.normal(0, 1, size=n)], axis=1)  # channel_1 == channel_2

    model = PCAReconstructor(n_components=1).fit(train)
    broken_row = np.array([[0.5, -0.5, 0.1]])  # correlation between channel 0/1 violated

    per_channel = model.score_per_channel(broken_row)
    assert per_channel.shape == (1, 3)

    top = model.most_anomalous_channels(broken_row, channel_names=["a", "b", "c"], top_n=2)
    assert set(top[0]) == {"a", "b"}  # the broken pair, not the untouched independent channel


def test_partial_fit_can_be_used_for_calibrate_and_predict_like_fit():
    rng = np.random.default_rng(5)
    train = rng.normal(0, 1, size=(2000, 4))
    val = rng.normal(0, 1, size=(500, 4))

    model = PCAReconstructor(n_components=0.95)
    model.partial_fit(train)
    model.calibrate_threshold(val, percentile=99.0)

    outlier = np.full((1, 4), 50.0)
    assert model.predict(outlier)[0]
