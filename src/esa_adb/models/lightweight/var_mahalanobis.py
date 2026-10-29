"""Lagged-ridge forecaster scored by the JOINT Mahalanobis distance of its residual
vector instead of the worst single channel.

Same closed-form, natively online forecaster as `LaggedRidgeForecaster` (inherited
unchanged); only the score differs. See `residual_mahalanobis` for why a joint score
can catch correlation-breaking anomalies that a per-channel max cannot.
"""
from __future__ import annotations

import numpy as np

from esa_adb.models.lightweight.lagged_ridge import LaggedRidgeForecaster
from esa_adb.models.lightweight.residual_mahalanobis import ResidualMahalanobis


class VARMahalanobis(LaggedRidgeForecaster):
    def __init__(self, n_lags: int = 3, ridge_lambda: float = 1.0, **maha_kwargs):
        super().__init__(n_lags=n_lags, ridge_lambda=ridge_lambda)
        self.maha = ResidualMahalanobis(**maha_kwargs)

    def _residuals(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        features, targets = self._design(X)
        resid = np.zeros_like(X)
        if features.shape[0]:
            resid[self.n_lags :] = targets - features @ self.weights_
        return resid

    def _accumulate_and_solve(self, X: np.ndarray) -> None:
        super()._accumulate_and_solve(X)
        if self.weights_ is not None:
            self.maha.update(self._residuals(X)[self.n_lags :])

    def fit(self, X: np.ndarray) -> "VARMahalanobis":
        self.maha = ResidualMahalanobis(self.maha.trim, self.maha.shrink, self.maha.blend)
        return super().fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("VARMahalanobis must be fit before scoring")
        s = self.maha.score(self._residuals(X))
        s[: self.n_lags] = 0.0
        return s
