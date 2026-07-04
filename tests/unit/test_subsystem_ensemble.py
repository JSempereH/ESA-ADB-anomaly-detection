import numpy as np

from esa_adb.models.subsystem_ensemble import SubsystemPCAEnsemble


def _correlated_group(rng, n, k, noise=0.01):
    latent = rng.normal(0, 1, size=n)
    return np.stack([latent + noise * rng.normal(size=n) for _ in range(k)], axis=1)


def test_small_groups_below_min_channels_are_dropped():
    ensemble = SubsystemPCAEnsemble({"sys_a": [0, 1, 2], "sys_b": [3, 4]}, min_channels=3)
    assert set(ensemble.channel_groups) == {"sys_a"}


def test_fit_creates_one_model_per_surviving_group():
    rng = np.random.default_rng(0)
    n = 1000
    X = np.concatenate([_correlated_group(rng, n, 3), _correlated_group(rng, n, 3)], axis=1)
    ensemble = SubsystemPCAEnsemble({"sys_a": [0, 1, 2], "sys_b": [3, 4, 5]}, min_channels=3).fit(X)

    assert set(ensemble.models) == {"sys_a", "sys_b"}


def test_predict_ors_across_subsystems():
    rng = np.random.default_rng(1)
    n = 2000
    train = np.concatenate([_correlated_group(rng, n, 3), _correlated_group(rng, n, 3)], axis=1)

    ensemble = SubsystemPCAEnsemble({"sys_a": [0, 1, 2], "sys_b": [3, 4, 5]}, min_channels=3, n_components=1).fit(train)
    ensemble.calibrate_threshold(train, percentile=99.0)

    normal_row = np.array([[0.1, 0.1, 0.1, 0.2, 0.2, 0.2]])
    # Break correlation only within sys_a; sys_b stays consistent with its own subspace.
    anomalous_in_sys_a = np.array([[5.0, -5.0, 5.0, 0.2, 0.2, 0.2]])

    assert not ensemble.predict(normal_row)[0]
    assert ensemble.predict(anomalous_in_sys_a)[0]


def test_partial_fit_self_initializes_subsystem_models():
    rng = np.random.default_rng(9)
    n = 2000
    X = np.concatenate([_correlated_group(rng, n, 3), _correlated_group(rng, n, 3)], axis=1)
    ensemble = SubsystemPCAEnsemble({"sys_a": [0, 1, 2], "sys_b": [3, 4, 5]}, min_channels=3)

    for start in range(0, n, 200):
        ensemble.partial_fit(X[start : start + 200])

    assert set(ensemble.models) == {"sys_a", "sys_b"}
    ensemble.calibrate_threshold(X, percentile=99.0)

    normal_row = np.array([[0.1, 0.1, 0.1, 0.2, 0.2, 0.2]])
    anomalous_in_sys_a = np.array([[5.0, -5.0, 5.0, 0.2, 0.2, 0.2]])
    assert not ensemble.predict(normal_row)[0]
    assert ensemble.predict(anomalous_in_sys_a)[0]


def test_min_agreeing_requires_multiple_subsystems_to_flag_the_same_row():
    rng = np.random.default_rng(7)
    n = 3000
    # Three independent subsystems -- each with its own latent factor.
    train = np.concatenate([_correlated_group(rng, n, 3) for _ in range(3)], axis=1)
    groups = {"a": [0, 1, 2], "b": [3, 4, 5], "c": [6, 7, 8]}
    ensemble = SubsystemPCAEnsemble(groups, min_channels=3, n_components=1).fit(train)
    ensemble.calibrate_threshold(train, percentile=99.0)

    normal_row = np.array([[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]])
    # Only subsystem "a"'s internal correlation is broken -- b and c stay consistent.
    single_subsystem_break = np.array([[5.0, -5.0, 5.0, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]])

    assert not ensemble.predict(normal_row, min_agreeing=1)[0]
    # OR rule (min_agreeing=1) flags a break confined to one subsystem...
    assert ensemble.predict(single_subsystem_break, min_agreeing=1)[0]
    # ...but requiring 2+ subsystems to agree correctly withholds it (no corroboration).
    assert not ensemble.predict(single_subsystem_break, min_agreeing=2)[0]


def test_score_per_subsystem_returns_one_array_per_group():
    rng = np.random.default_rng(2)
    n = 500
    X = np.concatenate([_correlated_group(rng, n, 3), _correlated_group(rng, n, 3)], axis=1)
    ensemble = SubsystemPCAEnsemble({"sys_a": [0, 1, 2], "sys_b": [3, 4, 5]}, min_channels=3).fit(X)

    scores = ensemble.score_per_subsystem(X)

    assert set(scores) == {"sys_a", "sys_b"}
    assert scores["sys_a"].shape == (n,)
