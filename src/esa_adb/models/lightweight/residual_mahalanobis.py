"""Joint Mahalanobis scoring of forecast-residual vectors, shared by the VAR and
echo-state detectors.

A per-channel residual score (what `LaggedRidgeForecaster` does) can only see a
channel that is wrong on its own. The residual VECTOR of a healthy spacecraft is
strongly correlated across channels, so an anomaly that breaks that correlation while
keeping every residual small is invisible per channel but large in Mahalanobis
distance. This is the same lever that made the Kalman block and the copula detectors
work, applied to forecast residuals.

The covariance is closed-form and online: each batch's covariance is blended 50/50 with
the previous one. Two guards against a contaminated fit batch, because at L2 the batch
contains anomalies: the highest-distance rows are trimmed before re-estimating, and the
result is shrunk towards its own diagonal.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class ResidualMahalanobis:
    def __init__(self, trim: float = 0.02, shrink: float = 0.05, blend: float = 0.5):
        self.trim = trim
        self.shrink = shrink
        self.blend = blend
        self.cov_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None

    @staticmethod
    def _d2(resid: np.ndarray, precision: np.ndarray) -> np.ndarray:
        return np.einsum("ti,ij,tj->t", resid, precision, resid)

    def update(self, resid: np.ndarray) -> "ResidualMahalanobis":
        resid = np.asarray(resid, dtype=np.float64)
        d = resid.shape[1]
        centered = resid - resid.mean(axis=0)
        cov = np.cov(centered.T) + 1e-12 * np.eye(d)
        d2 = self._d2(centered, np.linalg.pinv(cov))
        keep = d2 <= np.quantile(d2, 1.0 - self.trim)
        cov = np.cov(centered[keep].T)
        cov = (1.0 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov))
        self.cov_ = cov if self.cov_ is None else self.blend * self.cov_ + (1.0 - self.blend) * cov
        self.precision_ = np.linalg.pinv(self.cov_ + 1e-12 * np.eye(d))
        return self

    def score(self, resid: np.ndarray) -> np.ndarray:
        """sqrt(mean Mahalanobis^2 per dimension): ~1 for a nominal row at any d."""
        if self.precision_ is None:
            raise RuntimeError("ResidualMahalanobis must be updated before scoring")
        resid = np.asarray(resid, dtype=np.float64)
        return np.sqrt(np.maximum(self._d2(resid, self.precision_), 0.0) / resid.shape[1])
