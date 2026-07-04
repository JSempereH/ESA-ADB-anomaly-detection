import numpy as np
import polars as pl
import pytest

from esa_adb.evaluation import evaluate_window_predictions
from esa_adb.models.base import build_window_feature_matrix, window_scores_to_row_scores
from esa_adb.models.lightweight import GlobalSTD, OnlineGlobalSTD, WindowedIsolationForest
from esa_adb.preprocessing.windowing import WindowSpec


def _nominal_data(rng, n=2000, n_channels=3):
    return rng.normal(loc=0.0, scale=1.0, size=(n, n_channels))


def test_global_std_flags_outlier_channel():
    rng = np.random.default_rng(0)
    train = _nominal_data(rng)
    model = GlobalSTD(n_std=3.0).fit(train)

    normal_row = np.array([[0.1, -0.2, 0.05]])
    outlier_row = np.array([[10.0, 0.0, 0.0]])  # channel 0 wildly out of range

    assert not model.predict(normal_row)[0]
    assert model.predict(outlier_row)[0]
    assert model.score(outlier_row)[0] > model.score(normal_row)[0]


def test_global_std_most_anomalous_channels_identifies_the_outlier():
    rng = np.random.default_rng(0)
    train = _nominal_data(rng)
    model = GlobalSTD(n_std=3.0).fit(train)

    outlier_row = np.array([[10.0, 0.0, 0.0]])  # channel 0 wildly out of range
    per_channel = model.score_per_channel(outlier_row)
    assert per_channel.shape == (1, 3)

    top = model.most_anomalous_channels(outlier_row, channel_names=["a", "b", "c"], top_n=1)
    assert top == [["a"]]


def test_online_global_std_most_anomalous_channels():
    rng = np.random.default_rng(1)
    model = OnlineGlobalSTD(n_channels=3)
    model.fit(_nominal_data(rng))

    outlier_row = np.array([[0.0, -8.0, 0.0]])  # channel 1 wildly out of range
    top = model.most_anomalous_channels(outlier_row, channel_names=["a", "b", "c"], top_n=1)
    assert top == [["b"]]


def test_global_std_from_standardization_stats():
    stats = pl.DataFrame(
        {
            "channel": ["channel_1", "channel_2"],
            "mean": [0.0, 5.0],
            "std": [1.0, 2.0],
        }
    )
    model = GlobalSTD.from_standardization_stats(stats, ["channel_1", "channel_2"], n_std=3.0)

    assert model.mean_.tolist() == [0.0, 5.0]
    assert model.std_.tolist() == [1.0, 2.0]
    assert not model.predict(np.array([[0.5, 5.5]]))[0]
    assert model.predict(np.array([[10.0, 5.5]]))[0]


def test_global_std_constant_channel_no_div_by_zero():
    X = np.zeros((100, 2))
    X[:, 1] = np.random.default_rng(0).normal(size=100)
    model = GlobalSTD().fit(X)
    assert model.std_[0] == 1.0  # guarded, not 0
    scores = model.score(np.array([[0.0, 0.0]]))
    assert np.isfinite(scores).all()


def test_online_global_std_matches_batch_after_full_pass():
    rng = np.random.default_rng(1)
    X = _nominal_data(rng, n=500)

    online = OnlineGlobalSTD(n_channels=3)
    online.fit(X)
    batch = GlobalSTD().fit(X)

    assert online.mean_ == pytest.approx(batch.mean_, abs=1e-6)
    assert online.std_ == pytest.approx(batch.std_, abs=1e-6)


def test_online_global_std_partial_fit_incremental():
    online = OnlineGlobalSTD(n_channels=1)
    for x in [np.array([1.0]), np.array([2.0]), np.array([3.0])]:
        online.partial_fit(x)
    assert online.mean_[0] == pytest.approx(2.0)
    assert online.n == 3


def test_build_window_feature_matrix_shape():
    values = np.arange(20).reshape(10, 2).astype(float)
    spec = WindowSpec(length=4, stride=2)
    X = build_window_feature_matrix(values, spec)
    assert X.shape == (4, 8)  # (10-4)//2 + 1 = 4 windows, 4*2=8 features
    assert X[0].tolist() == values[0:4].reshape(-1).tolist()


def test_windowed_isolation_forest_separates_injected_outlier_window():
    rng = np.random.default_rng(42)
    n_channels = 2
    nominal = rng.normal(0, 1, size=(3000, n_channels))
    spec = WindowSpec(length=10, stride=10)

    model = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(nominal)

    normal_window = rng.normal(0, 1, size=(10, n_channels))
    anomalous_window = rng.normal(0, 1, size=(10, n_channels)) + 20.0  # shifted far away

    assert model.score(anomalous_window)[0] > model.score(normal_window)[0]


def test_windowed_isolation_forest_partial_fit_grows_forest_without_discarding_trees():
    rng = np.random.default_rng(7)
    n_channels = 2
    spec = WindowSpec(length=10, stride=10)

    model = WindowedIsolationForest(spec, n_estimators=50, contamination=0.05).fit(rng.normal(0, 1, size=(2000, n_channels)))
    first_batch_trees = list(model.model.estimators_)
    assert len(first_batch_trees) == 50

    model.partial_fit(rng.normal(0, 1, size=(2000, n_channels)), n_new_estimators=50)
    assert model.model.n_estimators == 100
    assert len(model.model.estimators_) == 100
    # the first 50 trees are the SAME fitted objects, not refit on the new batch
    assert model.model.estimators_[:50] == first_batch_trees


def test_windowed_isolation_forest_partial_fit_still_separates_outliers():
    rng = np.random.default_rng(8)
    n_channels = 2
    spec = WindowSpec(length=10, stride=10)

    model = WindowedIsolationForest(spec, n_estimators=50, contamination=0.05)
    model.partial_fit(rng.normal(0, 1, size=(1500, n_channels)))
    model.partial_fit(rng.normal(0, 1, size=(1500, n_channels)))

    normal_window = rng.normal(0, 1, size=(10, n_channels))
    anomalous_window = rng.normal(0, 1, size=(10, n_channels)) + 20.0
    assert model.score(anomalous_window)[0] > model.score(normal_window)[0]


def test_window_scores_to_row_scores_takes_max_over_overlap():
    spec = WindowSpec(length=4, stride=2)
    n_rows = 8  # windows: [0,4), [2,6), [4,8) -> indices 0,1,2
    window_scores = np.array([1.0, 5.0, 2.0])
    row_scores = window_scores_to_row_scores(window_scores, n_rows, spec)
    # row 3 is covered by window 0 (score 1.0) and window 1 (score 5.0) -> max = 5.0
    assert row_scores[3] == 5.0
    assert row_scores[0] == 1.0
    assert row_scores[7] == 2.0


def test_evaluate_window_predictions_precision_recall():
    labelled_bounds = pl.DataFrame(
        {
            "window_index": [0, 1, 2, 3],
            "is_nominal": [True, False, False, True],
        }
    )
    predicted = np.array([False, True, False, True])  # 1 TP, 1 FN, 1 FP, 1 TN

    result = evaluate_window_predictions(predicted, labelled_bounds, beta=1.0)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f_beta"] == pytest.approx(0.5)
