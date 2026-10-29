"""Dynamic factor model anomaly detector.

Plain PCA is a static factor model: each row's factors come from a fresh
projection, with no sense of how the factor values should move over time. This
version adds that: each factor gets its own simple AR(1) Kalman filter, so a row
can also be flagged if the shared, cross-channel component jumps in a way its own
history does not predict, even if every channel still looks fine individually.

Loadings are fit once per window with ordinary PCA, same as PCAReconstructor.
Each row's factors are estimated by direct projection. The score is the larger
of the usual PCA reconstruction error and the new factor-surprise score, so this
model can only add detection signal on top of plain PCA, never take it away.
That makes it a fair test of whether the extra factor dynamics actually help.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA

from esa_adb.models.base import AnomalyDetector


class DynamicFactorModel(AnomalyDetector):
    def __init__(
        self,
        n_components: Union[float, int] = 0.95,
        ar_clip: float = 0.98,
        initial_variance: float = 1e6,
    ) -> None:
        self.n_components = n_components
        self.ar_clip = ar_clip
        self.initial_variance = initial_variance
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None
        self.k_: Optional[int] = None
        self.phi_: Optional[np.ndarray] = None  # (k,) per-factor AR(1) coefficient
        self.Q_: Optional[np.ndarray] = None  # (k,) per-factor innovation variance
        self.factor_state: Optional[np.ndarray] = None  # (k,)
        self.factor_var: Optional[np.ndarray] = None  # (k,)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray) -> "DynamicFactorModel":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std

        X_std = self._standardize(X)
        self.pca = PCA(n_components=self.n_components, random_state=42).fit(X_std)
        self.k_ = self.pca.n_components_

        factors = self.pca.transform(X_std)  # (n, k), zero-mean by construction
        prev, curr = factors[:-1], factors[1:]
        denom = (prev**2).sum(axis=0)
        denom[denom == 0] = 1.0
        phi = (prev * curr).sum(axis=0) / denom
        self.phi_ = np.clip(phi, -self.ar_clip, self.ar_clip)
        resid = curr - self.phi_ * prev
        q = resid.var(axis=0, ddof=1)
        q[q == 0] = 1e-12
        self.Q_ = q

        self.factor_state = np.zeros(self.k_)
        self.factor_var = np.full(self.k_, self.initial_variance)
        self.score(X)  # warm up factor-filter state through the fit batch
        return self

    def partial_fit(self, X: np.ndarray) -> "DynamicFactorModel":
        if self.pca is None:
            return self.fit(X)
        self.score(np.asarray(X, dtype=np.float64))
        return self

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error per channel, same as PCAReconstructor. Does not
        include the factor-surprise score, which has no per-channel breakdown."""
        if self.pca is None:
            raise RuntimeError("DynamicFactorModel must be fit before scoring")
        X_std = self._standardize(np.asarray(X, dtype=np.float64))
        factors = self.pca.transform(X_std)
        recon = self.pca.inverse_transform(factors)
        return (X_std - recon) ** 2

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("DynamicFactorModel must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        X_std = self._standardize(X)
        factors = self.pca.transform(X_std)  # (n, k), direct per-row projection
        recon = self.pca.inverse_transform(factors)
        idio_score = np.sqrt(((X_std - recon) ** 2).mean(axis=1))

        n, k = factors.shape
        factor_innov_z = np.empty((n, k))
        for t in range(n):
            f_pred = self.phi_ * self.factor_state
            var_pred = (self.phi_**2) * self.factor_var + self.Q_
            innovation = factors[t] - f_pred
            factor_innov_z[t] = np.abs(innovation) / np.sqrt(var_pred)
            # The projection gives us the factor directly, with no observation
            # noise, so "update" just means adopting it as the new state.
            self.factor_state = factors[t]
            self.factor_var = self.Q_.copy()

        return np.maximum(idio_score, factor_innov_z.max(axis=1))
