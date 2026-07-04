import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.gdn import GDN, _build_graph  # noqa: E402


def _correlated_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    latent = np.sin(np.arange(n) * 0.05)
    c1 = latent + rng.normal(0, 0.05, size=n)
    c2 = latent + rng.normal(0, 0.05, size=n)  # correlated with c1
    c3 = rng.normal(0, 1, size=n)  # independent
    return np.stack([c1, c2, c3], axis=1)


def test_build_graph_always_includes_self_loops_and_respects_top_k():
    rng = np.random.default_rng(0)
    import torch as t

    embeddings = t.tensor(rng.normal(size=(5, 4)), dtype=t.float32)
    adjacency = _build_graph(embeddings, top_k=2)
    assert adjacency.shape == (5, 5)
    assert adjacency.diagonal().all()  # every node is its own neighbor
    off_diagonal_counts = adjacency.sum(dim=1) - 1  # exclude the self-loop
    assert (off_diagonal_counts == 2).all()


def test_fit_builds_graph_and_trains():
    X = _correlated_data()
    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    model.fit(X, epochs=5, batch_size=64, verbose=False)
    assert model.graph_built
    assert model.net.adjacency.diagonal().all()


def test_calibrate_threshold_and_predict():
    rng = np.random.default_rng(1)
    train = _correlated_data(seed=1)
    val = _correlated_data(seed=2)

    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    model.fit(train, epochs=10, batch_size=64, verbose=False)
    model.calibrate_threshold(val, percentile=99.0)

    assert model.threshold_ is not None
    outlier = np.tile(np.array([[50.0, -50.0, 50.0]]), (15, 1))  # break the c1/c2 correlation, far outside range
    combined = np.concatenate([val[-20:], outlier])
    preds = model.predict(combined)
    assert preds[-1]  # the injected outlier at the end should be flagged
    del rng


def test_score_before_calibration_raises():
    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    model.fit(_correlated_data(), epochs=2, batch_size=64, verbose=False)
    with pytest.raises(RuntimeError):
        model.score(_correlated_data(seed=3))


def test_fit_rebuilds_graph_from_trained_embeddings_not_just_random_init():
    # The graph must reflect trained embeddings, not just the random initialization --
    # with enough channels that top_k doesn't saturate into a fully-connected graph (so
    # there's actual room for the graph to change), fit() should produce a DIFFERENT
    # graph than the one its own random initial embeddings would have produced.
    rng = np.random.default_rng(20)
    n = 3000
    n_channels = 10
    latent = np.sin(np.arange(n) * 0.05)
    channels = [latent + rng.normal(0, 0.05, size=n) for _ in range(5)] + [rng.normal(0, 1, size=n) for _ in range(5)]
    X = np.stack(channels, axis=1)

    model = GDN(n_channels=n_channels, window_length=10, embed_dim=8, top_k=3, hidden_size=16, seed=99)
    initial_embeddings = model.net.sensor_embeddings.detach().clone()
    initial_graph = _build_graph(initial_embeddings, top_k=3)

    model.fit(X, epochs=15, batch_size=128, verbose=False)

    assert not torch.equal(model.net.sensor_embeddings, initial_embeddings)  # embeddings actually trained
    assert not torch.equal(model.net.adjacency, initial_graph)  # graph reflects that training, not just random init


def test_partial_fit_self_initializes_and_freezes_graph_after_first_call():
    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    assert not model.graph_built
    model.partial_fit(_correlated_data(seed=4), epochs=3, batch_size=64, verbose=False)
    assert model.graph_built
    adjacency_after_first = model.net.adjacency.clone()

    model.partial_fit(_correlated_data(seed=5), epochs=3, batch_size=64, verbose=False)
    torch.testing.assert_close(model.net.adjacency, adjacency_after_first)  # never rebuilt


def test_score_per_channel_shape_and_most_anomalous_channels():
    train = _correlated_data(seed=6)
    val = _correlated_data(seed=7)
    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    model.fit(train, epochs=10, batch_size=64, verbose=False)
    model.calibrate_threshold(val, percentile=99.0)

    per_channel = model.score_per_channel(val)
    assert per_channel.shape == (val.shape[0], 3)

    names = ["c1", "c2", "c3"]
    top = model.most_anomalous_channels(val[:5], channel_names=names, top_n=2)
    assert len(top) == 5
    assert all(len(row) == 2 for row in top)
    assert all(name in names for row in top for name in row)


def test_graph_edges_returns_channel_name_pairs_without_self_loops():
    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    model.fit(_correlated_data(seed=8), epochs=2, batch_size=64, verbose=False)

    edges = model.graph_edges(channel_names=["c1", "c2", "c3"])
    assert all(src != dst for src, dst in edges)
    assert all(src in {"c1", "c2", "c3"} and dst in {"c1", "c2", "c3"} for src, dst in edges)


def test_attention_for_window_shape_and_normalization():
    model = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16)
    model.fit(_correlated_data(seed=9), epochs=2, batch_size=64, verbose=False)

    window = _correlated_data(seed=10)[:10]
    attention = model.attention_for_window(window)
    assert attention.shape == (3, 3)
    # each row is a softmax over that channel's neighbors -- rows sum to 1
    np.testing.assert_allclose(attention.sum(axis=1), np.ones(3), atol=1e-5)


def test_same_seed_gives_identical_scores():
    X = _correlated_data(seed=11)
    a = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16, seed=123)
    a.fit(X, epochs=5, batch_size=64, verbose=False)
    a.calibrate_threshold(X, percentile=99.0)

    b = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16, seed=123)
    b.fit(X, epochs=5, batch_size=64, verbose=False)
    b.calibrate_threshold(X, percentile=99.0)

    np.testing.assert_array_equal(a.score(X), b.score(X))


def test_different_seed_gives_different_scores():
    X = _correlated_data(seed=12)
    a = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16, seed=1)
    a.fit(X, epochs=5, batch_size=64, verbose=False)
    a.calibrate_threshold(X, percentile=99.0)

    b = GDN(n_channels=3, window_length=10, embed_dim=8, top_k=2, hidden_size=16, seed=2)
    b.fit(X, epochs=5, batch_size=64, verbose=False)
    b.calibrate_threshold(X, percentile=99.0)

    assert not np.array_equal(a.score(X), b.score(X))
