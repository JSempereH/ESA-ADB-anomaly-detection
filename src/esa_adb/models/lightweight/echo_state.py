"""Echo-state (reservoir) forecaster with a closed-form online ridge readout, scored by
the joint Mahalanobis distance of its one-step residual.

The nonlinear counterpart of `VARMahalanobis`: a fixed random recurrent reservoir turns
the history of all channels into a state vector, and only the linear readout
(state, current input -> next input) is learned, by accumulating a Gram matrix exactly
like `LaggedRidgeForecaster`. No gradient descent, nothing forgotten, so it keeps the
online properties the gradient-trained forecasters lack.

The reservoir restarts from zero at every call (fit or score chunk) and the first
`washout` rows of a chunk score 0, because a chunk is a separate contiguous segment.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.lightweight.residual_mahalanobis import ResidualMahalanobis


class EchoStateForecaster(AnomalyDetector):
    def __init__(
        self, n_reservoir: int = 200, spectral_radius: float = 0.9, leak: float = 0.3,
        input_scale: float = 0.5, density: float = 0.1, ridge_lambda: float = 1e-3, washout: int = 100, seed: int = 42,
    ):
        self.n_reservoir, self.spectral_radius, self.leak = n_reservoir, spectral_radius, leak
        self.input_scale, self.density, self.ridge_lambda = input_scale, density, ridge_lambda
        self.washout, self.seed = washout, seed
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.w_in_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self._gram: Optional[np.ndarray] = None
        self._cross: Optional[np.ndarray] = None
        self._n_seen = 0
        self.weights_: Optional[np.ndarray] = None
        self.maha = ResidualMahalanobis()

    def _init(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        d, n = X.shape[1], self.n_reservoir
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self.w_in_ = rng.uniform(-self.input_scale, self.input_scale, size=(n, d))
        w = rng.normal(size=(n, n)) * (rng.random((n, n)) < self.density)
        radius = np.max(np.abs(np.linalg.eigvals(w))) or 1.0
        self.w_ = w * (self.spectral_radius / radius)

    def _states(self, Xs: np.ndarray) -> np.ndarray:
        n_rows = Xs.shape[0]
        drive = Xs @ self.w_in_.T
        states = np.empty((n_rows, self.n_reservoir))
        s = np.zeros(self.n_reservoir)
        a = self.leak
        for t in range(n_rows):
            s = (1.0 - a) * s + a * np.tanh(drive[t] + self.w_ @ s)
            states[t] = s
        return states

    def _design(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """features[t] = [1, state_t, x_t] -> target x_{t+1}, for t in [washout, n-1)."""
        Xs = np.clip((np.asarray(X, dtype=np.float64) - self.mean_) / self.std_, -10.0, 10.0)
        states = self._states(Xs)
        feats = np.hstack([np.ones((Xs.shape[0], 1)), states, Xs])
        return feats[self.washout : -1], Xs[self.washout + 1 :]

    def partial_fit(self, X: np.ndarray) -> "EchoStateForecaster":
        X = np.asarray(X, dtype=np.float64)
        if self.w_ is None:
            self._init(X)
        feats, targets = self._design(X)
        if self._gram is None:
            self._gram = np.zeros((feats.shape[1], feats.shape[1]))
            self._cross = np.zeros((feats.shape[1], targets.shape[1]))
        self._gram += feats.T @ feats
        self._cross += feats.T @ targets
        self._n_seen += feats.shape[0]
        reg = self.ridge_lambda * self._n_seen * np.eye(self._gram.shape[0])
        reg[0, 0] = 0.0
        self.weights_ = np.linalg.solve(self._gram + reg, self._cross)
        self.maha.update(targets - feats @ self.weights_)
        return self

    def fit(self, X: np.ndarray) -> "EchoStateForecaster":
        self.w_ = None
        self._gram = self._cross = None
        self._n_seen = 0
        self.maha = ResidualMahalanobis()
        return self.partial_fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("EchoStateForecaster must be fit before scoring")
        feats, targets = self._design(X)
        out = np.zeros(np.asarray(X).shape[0])
        # residual of predicting x_{t+1} is attributed to row t+1
        out[self.washout + 1 :] = self.maha.score(targets - feats @ self.weights_)
        return out
