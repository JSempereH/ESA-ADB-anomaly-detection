from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector


class LaggedRidgeForecaster(AnomalyDetector):
    """Linear multivariate forecasting-error detector: predict every channel's next
    value from the last `n_lags` samples of ALL channels via ridge regression, score by
    normalized forecast error.

    The cheap, honest counterpart to `LSTMForecaster`: recent forecasting literature
    repeatedly shows well-regularized linear models are competitive with far heavier
    sequence models (the DLinear line of work), and unlike the LSTM this trains in one
    closed-form solve. It is also *natively* online: `fit`/`partial_fit` only ever
    accumulate the sufficient statistics G = sum(x x^T), C = sum(x y^T) (a
    `(1+n_lags*n_channels)^2` Gram matrix -- 142x142 for Mission2's 47 channels at 3
    lags), so updating on a new month costs one pass over it plus one solve, and NOTHING
    about older months is ever discarded or approximated. This is exactly the
    "online training" (OT) property the 2024 Correia et al. survey flags as rare.

    `score()` is the max over channels of |forecast error| / per-channel residual std
    (worst-offending channel drives the row, same convention as `GlobalSTD`), with the
    first `n_lags` rows scoring 0 (no context yet -- `LSTMForecaster`'s convention).
    Residual stds are estimated from the most recent fit batch under the CURRENT
    weights, blended 50/50 with the previous estimate so one odd month can't redefine
    a channel's noise floor outright.
    """

    def __init__(self, n_lags: int = 3, ridge_lambda: float = 1.0):
        self.n_lags = n_lags
        self.ridge_lambda = ridge_lambda
        self._gram: Optional[np.ndarray] = None  # (d, d), d = 1 + n_lags*n_channels
        self._cross: Optional[np.ndarray] = None  # (d, n_channels)
        self._n_seen = 0
        self.weights_: Optional[np.ndarray] = None
        self.resid_std_: Optional[np.ndarray] = None

    def _design(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(features, targets): features[t] = [1, X[t-p], ..., X[t-1]] flattened,
        targets[t] = X[t], for t in [n_lags, n_rows)."""
        p = self.n_lags
        n_rows, n_channels = X.shape
        if n_rows <= p:
            d = 1 + p * n_channels
            return np.empty((0, d)), np.empty((0, n_channels))
        windows = np.lib.stride_tricks.sliding_window_view(X, (p, n_channels))[:-1]  # (n_rows-p, 1, p, n_channels)
        features = windows.reshape(n_rows - p, p * n_channels)
        features = np.hstack([np.ones((features.shape[0], 1)), features])
        return features, X[p:]

    def _accumulate_and_solve(self, X: np.ndarray) -> None:
        features, targets = self._design(np.asarray(X, dtype=np.float64))
        if features.shape[0] == 0:
            return
        if self._gram is None:
            d = features.shape[1]
            self._gram = np.zeros((d, d))
            self._cross = np.zeros((d, X.shape[1]))
        self._gram += features.T @ features
        self._cross += features.T @ targets
        self._n_seen += features.shape[0]
        # Effective lambda scales with samples seen, so regularization strength relative
        # to the data stays constant as the accumulated Gram matrix grows.
        reg = self.ridge_lambda * self._n_seen * np.eye(self._gram.shape[0])
        reg[0, 0] = 0.0  # never penalize the intercept
        self.weights_ = np.linalg.solve(self._gram + reg, self._cross)

        resid = targets - features @ self.weights_
        batch_std = resid.std(axis=0, ddof=1)
        batch_std[batch_std == 0] = 1.0
        if self.resid_std_ is None:
            self.resid_std_ = batch_std
        else:
            self.resid_std_ = np.sqrt(0.5 * self.resid_std_**2 + 0.5 * batch_std**2)

    def fit(self, X: np.ndarray) -> "LaggedRidgeForecaster":
        self._gram = None
        self._cross = None
        self._n_seen = 0
        self.resid_std_ = None
        self._accumulate_and_solve(X)
        return self

    def partial_fit(self, X: np.ndarray) -> "LaggedRidgeForecaster":
        self._accumulate_and_solve(X)
        return self

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        """`(n_rows, n_channels)` |forecast error| in residual-std units -- same
        explainability primitive as every other detector here."""
        if self.weights_ is None:
            raise RuntimeError("LaggedRidgeForecaster must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        features, targets = self._design(X)
        z = np.zeros_like(X)
        if features.shape[0]:
            z[self.n_lags :] = np.abs(targets - features @ self.weights_) / self.resid_std_
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
