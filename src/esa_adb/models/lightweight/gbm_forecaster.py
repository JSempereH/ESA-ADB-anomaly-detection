from __future__ import annotations

from typing import Optional

import numpy as np
from lightgbm import LGBMRegressor

from esa_adb.models.base import AnomalyDetector


class GBMForecaster(AnomalyDetector):
    """Forecasts each channel from recent lags of every channel, one gradient-
    boosted tree per target channel, and scores by normalized forecast error.
    The nonlinear counterpart to LaggedRidgeForecaster.

    Gradient-boosted trees have no exact incremental update the way a closed-form
    regression does. partial_fit keeps a rolling buffer of recent rows and
    retrains from that buffer each call, real memory but bounded, not an exact
    running fit.
    """

    def __init__(
        self,
        n_lags: int = 3,
        n_estimators: int = 20,
        num_leaves: int = 7,
        learning_rate: float = 0.15,
        max_buffer_rows: int = 100_000,
        seed: int = 42,
    ):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_buffer_rows = max_buffer_rows
        self.seed = seed
        self._buffer: Optional[np.ndarray] = None
        self.models_: Optional[list] = None
        self.resid_std_: Optional[np.ndarray] = None

    def _design(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Builds (features, targets): features[t] is the last n_lags rows before
        t flattened, targets[t] is X[t]. No intercept column, trees don't need one."""
        p = self.n_lags
        n_rows, n_channels = X.shape
        if n_rows <= p:
            return np.empty((0, p * n_channels)), np.empty((0, n_channels))
        windows = np.lib.stride_tricks.sliding_window_view(X, (p, n_channels))[:-1]
        features = windows.reshape(n_rows - p, p * n_channels)
        return features, X[p:]

    def _refit_from_buffer(self) -> None:
        X = self._buffer
        features, targets = self._design(X)
        if features.shape[0] == 0:
            return
        n_channels = targets.shape[1]
        models = []
        preds = np.zeros_like(targets)
        for c in range(n_channels):
            model = LGBMRegressor(
                n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, n_jobs=1, verbose=-1, random_state=self.seed,
            )
            model.fit(features, targets[:, c])
            preds[:, c] = model.predict(features)
            models.append(model)
        self.models_ = models

        batch_std = (targets - preds).std(axis=0, ddof=1)
        batch_std[batch_std == 0] = 1.0
        if self.resid_std_ is None:
            self.resid_std_ = batch_std
        else:
            self.resid_std_ = np.sqrt(0.5 * self.resid_std_**2 + 0.5 * batch_std**2)

    def _extend_buffer(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        self._buffer = X if self._buffer is None else np.concatenate([self._buffer, X], axis=0)
        if self._buffer.shape[0] > self.max_buffer_rows:
            self._buffer = self._buffer[-self.max_buffer_rows :]

    def fit(self, X: np.ndarray) -> "GBMForecaster":
        self._buffer = None
        self.resid_std_ = None
        self._extend_buffer(X)
        self._refit_from_buffer()
        return self

    def partial_fit(self, X: np.ndarray) -> "GBMForecaster":
        self._extend_buffer(X)
        self._refit_from_buffer()
        return self

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        if self.models_ is None:
            raise RuntimeError("GBMForecaster must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        features, targets = self._design(X)
        z = np.zeros_like(X)
        if features.shape[0]:
            preds = np.column_stack([m.predict(features) for m in self.models_])
            z[self.n_lags :] = np.abs(targets - preds) / self.resid_std_
        return z

    def most_anomalous_channels(self, X: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        z = self.score_per_channel(X)
        top_n = min(top_n, z.shape[1])
        order = np.argsort(-z, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.score_per_channel(X).max(axis=1)

    def predict(self, X: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        return self.score(X) > threshold
