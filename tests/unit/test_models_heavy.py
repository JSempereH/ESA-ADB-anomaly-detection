import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.heavy.lstm_forecaster import LSTMForecaster, build_forecast_windows  # noqa: E402


def test_build_forecast_windows_shapes():
    values = np.arange(20).reshape(10, 2).astype(float)
    X, y = build_forecast_windows(values, input_length=4, horizon=1, stride=1)
    assert X.shape == (6, 4, 2)  # 10 - 4 - 1 + 1 = 6 windows
    assert y.shape == (6, 1, 2)
    assert X[0].tolist() == values[0:4].tolist()
    assert y[0, 0].tolist() == values[4].tolist()


def test_build_forecast_windows_empty_when_too_short():
    values = np.zeros((3, 2))
    X, y = build_forecast_windows(values, input_length=4, horizon=1)
    assert X.shape[0] == 0
    assert X.shape[1:] == (4, 2)


def test_lstm_forecaster_learns_periodic_signal_and_flags_injected_anomaly():
    rng = np.random.default_rng(0)
    t = np.arange(2000)
    channel_1 = np.sin(t * 0.1)
    channel_2 = np.cos(t * 0.1)
    nominal = np.stack([channel_1, channel_2], axis=1) + rng.normal(0, 0.01, size=(2000, 2))

    model = LSTMForecaster(n_channels=2, input_length=20, hidden_size=16, num_layers=1)
    model.fit(nominal, epochs=15, batch_size=64, verbose=False)

    # A held-out nominal continuation should have low residual...
    t2 = np.arange(2000, 2100)
    nominal_continuation = np.stack([np.sin(t2 * 0.1), np.cos(t2 * 0.1)], axis=1)
    nominal_scores = model.score(nominal_continuation)[model.input_length :]

    # ...while an abrupt level-shift anomaly should have a much higher one.
    anomalous = nominal_continuation.copy()
    anomalous[50:] += 5.0
    anomalous_scores = model.score(anomalous)[model.input_length :]

    assert anomalous_scores.max() > nominal_scores.max() * 3

    preds = model.predict(anomalous)
    assert preds[: model.input_length].any() == np.False_  # leading gap never flagged
    assert preds[model.input_length + 50 :].any()  # the shifted region gets flagged


def test_threshold_calibration_uses_validation_not_training_residuals():
    # Calibrating the threshold from training residuals (where the model has directly
    # minimized error) underestimates real-world forecasting error and flags almost
    # everything on unseen, genuinely nominal data. Train is clean; a later, still-nominal
    # period is noisier (mimicking a different real mission phase).
    rng = np.random.default_rng(1)
    t_train = np.arange(1500)
    train = np.stack([np.sin(t_train * 0.1), np.cos(t_train * 0.1)], axis=1) + rng.normal(0, 0.01, size=(1500, 2))
    t_val = np.arange(1500, 1800)
    val = np.stack([np.sin(t_val * 0.1), np.cos(t_val * 0.1)], axis=1) + rng.normal(0, 0.2, size=(300, 2))

    with_val = LSTMForecaster(n_channels=2, input_length=20, hidden_size=16, num_layers=1)
    with_val.fit(train, validation_values=val, epochs=15, batch_size=64, verbose=False)

    train_only = LSTMForecaster(n_channels=2, input_length=20, hidden_size=16, num_layers=1)
    train_only.fit(train, epochs=15, batch_size=64, verbose=False)

    assert with_val.threshold_ > train_only.threshold_
    # The train-only-calibrated threshold over-flags genuinely nominal validation data;
    # the properly-calibrated one flags far less of it.
    assert train_only.predict(val).mean() > with_val.predict(val).mean()


def test_score_per_channel_and_most_anomalous_channels_identify_the_shifted_channel():
    rng = np.random.default_rng(5)
    t = np.arange(2000)
    channel_1 = np.sin(t * 0.1)
    channel_2 = np.cos(t * 0.1)
    channel_3 = np.sin(t * 0.05 + 1.0)  # never perturbed below
    nominal = np.stack([channel_1, channel_2, channel_3], axis=1) + rng.normal(0, 0.01, size=(2000, 3))

    model = LSTMForecaster(n_channels=3, input_length=20, hidden_size=16, num_layers=1)
    model.fit(nominal, epochs=15, batch_size=64, verbose=False)

    t2 = np.arange(2000, 2100)
    anomalous = np.stack([np.sin(t2 * 0.1), np.cos(t2 * 0.1), np.sin(t2 * 0.05 + 1.0)], axis=1)
    anomalous[50:, 0] += 5.0  # only channel_1 shifts

    per_channel = model.score_per_channel(anomalous)
    assert per_channel.shape == (100, 3)

    top = model.most_anomalous_channels(anomalous[model.input_length + 50 :], channel_names=["c1", "c2", "c3"], top_n=1)
    assert all(row == ["c1"] for row in top)


def test_partial_fit_continues_training_without_touching_threshold():
    rng = np.random.default_rng(4)
    t = np.arange(1500)
    batch1 = np.stack([np.sin(t * 0.1), np.cos(t * 0.1)], axis=1) + rng.normal(0, 0.01, size=(1500, 2))
    batch2 = np.stack([np.sin((t + 1500) * 0.1), np.cos((t + 1500) * 0.1)], axis=1) + rng.normal(0, 0.01, size=(1500, 2))

    model = LSTMForecaster(n_channels=2, input_length=20, hidden_size=16, num_layers=1)
    model.partial_fit(batch1, epochs=10, batch_size=64, verbose=False)
    assert model.threshold_ is None  # partial_fit never calibrates on its own

    model.partial_fit(batch2, epochs=10, batch_size=64, verbose=False)
    model.calibrate_threshold(batch2, percentile=99.0)
    assert model.threshold_ is not None
    # the model should have actually learned something -- residuals on the (noiseless-ish)
    # periodic signal it was trained on should be small relative to a wildly wrong forecast.
    assert model.score(batch2)[model.input_length :].mean() < 1.0


def test_residuals_chunked_scoring_matches_unchunked():
    # score()/_residuals() processes long inputs in row-chunks so scoring a full mission
    # doesn't try to materialize every sliding window in memory at once. Chunk boundaries
    # must not change the result versus one unchunked pass.
    rng = np.random.default_rng(2)
    t = np.arange(3000)
    values = np.stack([np.sin(t * 0.05), np.cos(t * 0.05)], axis=1) + rng.normal(0, 0.05, size=(3000, 2))

    model = LSTMForecaster(n_channels=2, input_length=15, hidden_size=8, num_layers=1)
    model.fit(values, epochs=2, batch_size=128, verbose=False)

    unchunked = model._residuals(values, chunk_rows=10_000)  # bigger than n -> single chunk
    chunked = model._residuals(values, chunk_rows=137)  # deliberately awkward chunk size

    np.testing.assert_allclose(chunked, unchunked, atol=1e-5)


def test_lstm_forecaster_raises_if_too_little_data():
    model = LSTMForecaster(n_channels=2, input_length=50)
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, 2)), epochs=1, verbose=False)
