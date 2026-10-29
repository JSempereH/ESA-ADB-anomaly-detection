import numpy as np

from esa_adb.models.lightweight.kalman_block_multivariate import KalmanBlockMultivariate


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    groups = {"a": [0, 1, 2], "b": [3, 4]}
    model = KalmanBlockMultivariate(channel_groups=groups).fit(X)
    scores = model.score(rng.normal(size=(100, 6)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_falls_back_to_per_channel_score_for_unblocked_channels():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    groups = {"a": [0, 1]}  # channels 2, 3 unblocked
    model = KalmanBlockMultivariate(channel_groups=groups, min_block_size=2).fit(X)
    assert list(model._unblocked) == [2, 3]
    scores = model.score(rng.normal(size=(50, 4)))
    assert np.isfinite(scores).all()


def test_detects_broken_cross_channel_correlation_independent_filters_would_miss():
    """The core hypothesis this model tests: two correlated channels (b = 2*a + noise)
    behaving individually-unremarkably but breaking their NORMAL relationship should
    score higher under the block-Mahalanobis score than independent per-channel scoring
    would (each channel alone stays within its own normal range)."""
    rng = np.random.default_rng(2)
    n = 2000
    a = rng.normal(size=n)
    b = 2 * a + rng.normal(scale=0.1, size=n)
    X = np.column_stack([a, b])
    groups = {"ab": [0, 1]}
    model = KalmanBlockMultivariate(channel_groups=groups, min_block_size=2).fit(X)

    # Correlation-consistent test rows (same a->b relationship): should score low.
    a_test = rng.normal(size=20)
    consistent = np.column_stack([a_test, 2 * a_test + rng.normal(scale=0.1, size=20)])

    # Correlation-BROKEN test rows: each channel individually within its usual range,
    # but b no longer follows its usual 2x relationship to a.
    broken = np.column_stack([a_test, -2 * a_test + rng.normal(scale=0.1, size=20)])

    consistent_scores = model.score(consistent)
    broken_scores = model.score(broken)
    assert broken_scores.mean() > consistent_scores.mean()


def test_min_block_size_excludes_small_groups():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3))
    groups = {"tiny": [0, 1]}
    model = KalmanBlockMultivariate(channel_groups=groups, min_block_size=3).fit(X)
    assert model._blocks == []
    assert list(model._unblocked) == [0, 1, 2]


def test_partial_fit_continues_state_without_resetting():
    rng = np.random.default_rng(4)
    groups = {"a": [0, 1]}
    model = KalmanBlockMultivariate(channel_groups=groups).fit(rng.normal(size=(300, 2)))
    state_before = model.state.copy()
    model.partial_fit(rng.normal(size=(100, 2)))
    assert not np.allclose(state_before, model.state)
