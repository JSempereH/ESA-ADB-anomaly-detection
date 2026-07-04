import numpy as np
import pytest

from esa_adb.models.lightweight.half_space_trees import HalfSpaceTrees


def _nominal(n, n_channels=3, seed=0):
    return np.random.default_rng(seed).normal(0, 1, size=(n, n_channels))


def test_fit_builds_expected_tree_and_mass_array_shapes():
    model = HalfSpaceTrees(n_channels=3, n_trees=10, max_depth=5, window_size=50, seed=0).fit(_nominal(200))
    assert model.feature.shape == (10, 2**5 - 1)
    assert model.ref_mass.shape == (10, 2**5)
    assert model.ref_mass.sum() > 0  # at least one window swap happened over 200 rows / window_size 50


def test_score_shape_matches_input_rows():
    model = HalfSpaceTrees(n_channels=3, n_trees=10, max_depth=5, window_size=50, seed=0).fit(_nominal(200))
    scores = model.score(_nominal(30, seed=1))
    assert scores.shape == (30,)


def test_far_outlier_scores_higher_than_nominal_region():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, size=(3000, 3))
    model = HalfSpaceTrees(n_channels=3, n_trees=30, max_depth=8, window_size=200, seed=2).fit(train)

    nominal_row = np.array([[0.1, -0.1, 0.05]])
    outlier_row = np.array([[50.0, -50.0, 50.0]])

    assert model.score(outlier_row)[0] > model.score(nominal_row)[0]


def test_partial_fit_updates_mass_without_rebuilding_tree_structure():
    rng = np.random.default_rng(3)
    model = HalfSpaceTrees(n_channels=2, n_trees=5, max_depth=6, window_size=100, seed=3).fit(rng.normal(size=(200, 2)))
    feature_before = model.feature.copy()
    mass_before = model.ref_mass.copy()

    model.partial_fit(rng.normal(size=(150, 2)))

    np.testing.assert_array_equal(model.feature, feature_before)  # structure untouched
    assert not np.array_equal(model.ref_mass, mass_before)  # mass profile did change


def test_partial_fit_requires_prior_fit():
    model = HalfSpaceTrees(n_channels=2, seed=0)
    with pytest.raises(RuntimeError):
        model.partial_fit(np.zeros((5, 2)))


def test_partial_fit_batch_matches_row_by_row_streaming():
    # Feeding one big batch through partial_fit should land on the exact same window-swap
    # boundaries (and therefore the exact same final mass profile) as feeding the same
    # rows one at a time -- the chunking in partial_fit must not depend on caller batch size.
    rng = np.random.default_rng(9)
    X = rng.normal(size=(437, 2))  # deliberately not a multiple of window_size
    a = HalfSpaceTrees(n_channels=2, n_trees=4, max_depth=5, window_size=50, seed=1).fit(X[:1])
    for row in X[1:]:
        a.partial_fit(row[None, :])

    b = HalfSpaceTrees(n_channels=2, n_trees=4, max_depth=5, window_size=50, seed=1).fit(X[:1])
    b.partial_fit(X[1:])

    np.testing.assert_array_equal(a.ref_mass, b.ref_mass)
    np.testing.assert_array_equal(a.latest_mass, b.latest_mass)


def test_flush_window_captures_a_whole_batch_regardless_of_window_size():
    # window_size larger than the whole batch means partial_fit's own auto-swap never
    # fires -- flush_window() is the only thing that promotes latest_mass to ref_mass,
    # and it should capture the FULL batch, not just a tail slice.
    rng = np.random.default_rng(8)
    X = rng.normal(size=(5000, 3))
    model = HalfSpaceTrees(n_channels=3, n_trees=5, max_depth=6, window_size=10_000, seed=1).fit(X)

    assert model.ref_mass.sum() == 0  # no auto-swap happened yet (5000 < window_size)
    model.flush_window()
    assert model.ref_mass.sum() == 5000 * 5  # every row counted once per tree


def test_flush_window_requires_prior_fit():
    model = HalfSpaceTrees(n_channels=2, seed=0)
    with pytest.raises(RuntimeError):
        model.flush_window()


def test_same_seed_gives_identical_scores():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 3))
    a = HalfSpaceTrees(n_channels=3, n_trees=15, max_depth=6, window_size=100, seed=42).fit(X)
    b = HalfSpaceTrees(n_channels=3, n_trees=15, max_depth=6, window_size=100, seed=42).fit(X)
    np.testing.assert_array_equal(a.score(X), b.score(X))


def test_different_seed_gives_different_scores():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 3))
    a = HalfSpaceTrees(n_channels=3, n_trees=15, max_depth=6, window_size=100, seed=1).fit(X)
    b = HalfSpaceTrees(n_channels=3, n_trees=15, max_depth=6, window_size=100, seed=2).fit(X)
    assert not np.array_equal(a.score(X), b.score(X))


def test_predict_without_threshold_raises():
    model = HalfSpaceTrees(n_channels=2, seed=0).fit(_nominal(200, n_channels=2))
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 2)))


def test_predict_with_threshold():
    rng = np.random.default_rng(6)
    train = rng.normal(0, 1, size=(2000, 3))
    model = HalfSpaceTrees(n_channels=3, n_trees=20, max_depth=8, window_size=200, seed=6).fit(train)

    scores = model.score(train)
    thr = np.percentile(scores, 99.0)
    outlier = np.array([[50.0, -50.0, 50.0]])
    assert model.predict(outlier, threshold=thr)[0]
