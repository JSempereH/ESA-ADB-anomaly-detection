from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector


class _LSTMForecastNet(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int, num_layers: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_channels * horizon)
        self.n_channels = n_channels
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.view(-1, self.horizon, self.n_channels)


def build_forecast_windows(
    values: np.ndarray, input_length: int, horizon: int = 1, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) supervised pairs: X[i] = `input_length` rows of history, y[i] = the next
    `horizon` rows to forecast, for every valid window start (stepped by `stride`).
    """
    n = values.shape[0]
    X, y = [], []
    i = 0
    while i + input_length + horizon <= n:
        X.append(values[i : i + input_length])
        y.append(values[i + input_length : i + input_length + horizon])
        i += stride
    if not X:
        n_channels = values.shape[1]
        return np.empty((0, input_length, n_channels)), np.empty((0, horizon, n_channels))
    return np.stack(X), np.stack(y)


class LSTMForecaster(AnomalyDetector):
    """Multivariate LSTM forecasting anomaly detector, in the spirit of the paper's
    Telemanom-ESA (the strongest overall algorithm in ESA-ADB, especially for Mission1's
    full channel set) -- but forecasting every target channel from a single shared model,
    like the paper's own Telemanom-ESA extension over the original NASA Telemanom, whose
    one-model-per-output-channel design "can take months on a single PC" to scale to a
    full mission's worth of channels.

    `fit()` trains on nominal windows only (semi-supervised, matching the paper's own
    Telemanom-ESA training regime). `score()` returns one value per input row: the mean
    absolute forecasting error across channels between the model's one-step-ahead
    prediction and the real value, using every prior `input_length` rows as context --
    the same anomaly-score convention the paper's Telemanom-ESA itself uses. The first
    `input_length` rows of any call to `score()` have no prior context to forecast from
    and are scored 0 (never flagged), mirroring the "leading gap" idea already used in
    `preprocessing/resample.py`.

    Meaningfully training this (unlike the lightweight models) benefits from a GPU and
    more than a few minutes -- keep `epochs`/data volume small for local smoke-testing.
    """

    def __init__(
        self,
        n_channels: int,
        input_length: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 1,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.n_channels = n_channels
        self.input_length = input_length
        self.horizon = horizon
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)  # weight init draws from the global RNG at construction time
        self.net = _LSTMForecastNet(n_channels, hidden_size, num_layers, horizon).to(device)
        self.residual_std_: Optional[float] = None
        self.threshold_: Optional[float] = None

    def fit(
        self,
        values: np.ndarray,
        validation_values: Optional[np.ndarray] = None,
        threshold_percentile: float = 99.0,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        stride: int = 1,
        verbose: bool = True,
    ) -> "LSTMForecaster":
        """`values` should be nominal training data. Pass a separate, nominal
        `validation_values` slice (e.g. the mission's `val` split) to calibrate the
        default detection threshold -- using training residuals for this instead is a
        real trap: the model has directly minimized error on `values`, so its residual
        there systematically *underestimates* real-world forecasting error and yields a
        threshold so tight it flags almost every point on unseen data. Without
        `validation_values`, this falls back to training residuals with a printed
        warning, but that's a smoke-test convenience, not a recommended production setup.

        The threshold itself is a percentile of the (validation) residual distribution,
        not a Gaussian n-sigma assumption -- aggregate multi-channel forecasting error is
        often heavy-tailed, and GlobalSTD's own real-data validation showed exactly how
        badly a Gaussian assumption can fail on this kind of data.
        """
        self._train(values, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)

        if validation_values is None:
            if verbose:
                print(
                    "[LSTMForecaster] WARNING: no validation_values given -- calibrating "
                    "the threshold from training residuals, which underestimates real error."
                )
            calibration_values = values
        else:
            calibration_values = validation_values
        calibration_residuals = self._residuals(calibration_values)
        calibration_residuals = calibration_residuals[calibration_residuals > 0]  # drop the leading-gap zeros
        if calibration_residuals.size == 0:
            self.residual_std_, self.threshold_ = 1.0, 1.0
        else:
            self.residual_std_ = float(calibration_residuals.std())
            self.threshold_ = float(np.percentile(calibration_residuals, threshold_percentile))
        return self

    def partial_fit(
        self, values: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3, stride: int = 1, verbose: bool = True
    ) -> "LSTMForecaster":
        """Continue gradient descent on `self.net`'s EXISTING weights using only this new
        batch -- `fit()` already never reinitializes `self.net`, so repeated `fit()` calls
        already accumulate this way (unlike `Autoencoder`/`PCAReconstructor`, there's no
        per-call normalization step here to accidentally reset). This method exists
        mainly so callers building a walk-forward loop across several detector types can
        call the same `partial_fit(values)` on all of them -- it skips `fit()`'s bundled
        threshold calibration entirely (no `validation_values`/`threshold_percentile`
        params), leaving that to a separate `calibrate_threshold`-style call against each
        fold's own held-out data, same as `PCAReconstructor`/`HalfSpaceTrees`.
        """
        self._train(values, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "LSTMForecaster":
        """Set `threshold_`/`residual_std_` from a percentile of held-out nominal
        residuals, mirroring `PCAReconstructor.calibrate_threshold` -- the counterpart to
        `partial_fit` skipping `fit()`'s bundled calibration.
        """
        residuals = self._residuals(validation_values)
        residuals = residuals[residuals > 0]  # drop the leading-gap zeros
        if residuals.size == 0:
            self.residual_std_, self.threshold_ = 1.0, 1.0
        else:
            self.residual_std_ = float(residuals.std())
            self.threshold_ = float(np.percentile(residuals, percentile))
        return self

    def _train(self, values: np.ndarray, epochs: int, batch_size: int, lr: float, stride: int, verbose: bool) -> None:
        X, y = build_forecast_windows(values, self.input_length, self.horizon, stride)
        if X.shape[0] == 0:
            raise ValueError(
                f"Not enough rows ({values.shape[0]}) to build a single training window "
                f"of input_length={self.input_length} + horizon={self.horizon}"
            )
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n = X_t.shape[0]
        self.net.train()
        # Reseed here too (not just __init__): makes the epoch-by-epoch shuffling below
        # reproducible on its own, regardless of what other torch code ran between
        # construction and this call and may have advanced the global RNG state.
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for epoch in range(epochs):
            perm = torch.randperm(n, generator=generator)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                pred = self.net(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.shape[0]
            if verbose:
                print(f"[LSTMForecaster] epoch {epoch + 1}/{epochs} loss={total_loss / n:.5f}")

    def _channel_residuals(self, values: np.ndarray, chunk_rows: int = 20_000) -> np.ndarray:
        """`(n_rows, n_channels)` absolute forecast error per channel, processed in
        row-chunks rather than materializing every sliding window over `values` at once.
        Scoring a full mission's test split (millions of rows) as a single
        `build_forecast_windows` call would try to allocate an array of roughly
        `n_rows * input_length * n_channels` floats -- for Mission2's ~3M-row test split
        that's tens of gigabytes. Each chunk is padded with `input_length - 1` rows of
        preceding context so windows starting at a chunk boundary are still built
        correctly, mirroring the carry-over approach used for chunked resampling in
        `preprocessing/resample.py`. The explainability primitive `_residuals`'s
        row-level MAE (and therefore `score()`) averages over channels.
        """
        self.net.eval()
        n = values.shape[0]
        errors = np.zeros((n, self.n_channels))
        if n <= self.input_length + self.horizon - 1:
            return errors

        chunk_start = 0
        with torch.no_grad():
            while chunk_start < n:
                chunk_end = min(chunk_start + chunk_rows, n)
                ctx_start = max(0, chunk_start - self.input_length)
                X, y = build_forecast_windows(values[ctx_start:chunk_end], self.input_length, self.horizon, stride=1)
                if X.shape[0] > 0:
                    X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
                    pred = self.net(X_t).cpu().numpy()
                    err = np.abs(pred[:, 0, :] - y[:, 0, :])
                    out_start = ctx_start + self.input_length
                    errors[out_start : out_start + err.shape[0]] = err
                chunk_start = chunk_end
        return errors

    def _residuals(self, values: np.ndarray, chunk_rows: int = 20_000) -> np.ndarray:
        """Row-level forecast error (mean absolute error across channels) -- see
        `_channel_residuals` for the per-channel version this aggregates.
        """
        return self._channel_residuals(values, chunk_rows=chunk_rows).mean(axis=1)

    def score_per_channel(self, values: np.ndarray) -> np.ndarray:
        """`(n_rows, n_channels)` absolute forecast error per channel -- which channel(s)
        the model failed to predict well, same explainability role as
        `PCAReconstructor`/`Autoencoder`'s `score_per_channel`. Rows before
        `input_length` (no context to forecast from yet) score 0 on every channel, same
        as `score()`'s own leading-gap convention.
        """
        return self._channel_residuals(values)

    def most_anomalous_channels(self, values: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        """For each row, the `top_n` channels with the highest forecast error."""
        err = self.score_per_channel(values)
        top_n = min(top_n, err.shape[1])
        order = np.argsort(-err, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def score(self, values: np.ndarray) -> np.ndarray:
        return self._residuals(values)

    def predict(self, values: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        if threshold is not None:
            thr = threshold
        elif self.threshold_ is not None:
            thr = self.threshold_
        else:
            thr = 3.0 * (self.residual_std_ or 1.0)
        return self.score(values) > thr
