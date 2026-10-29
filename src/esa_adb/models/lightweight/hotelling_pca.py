"""Classical multivariate statistical process control on a PCA subspace: Hotelling's
T^2 inside the retained subspace plus the squared prediction error (SPE/Q) outside it.

`PCAReconstructor` only scores the SPE (distance from the subspace). T^2 adds the
complementary signal -- a row that stays on the subspace but sits far out along the
low-variance principal directions, which is exactly where a correlation-preserving but
unusually extreme state ends up. Each statistic is divided by its own 99th percentile on
the fit batch (blended online) so neither dominates, and the row score is the max.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.lightweight.pca_reconstruction import PCAReconstructor


class HotellingPCA(PCAReconstructor):
    def __init__(self, n_components=0.95, ref_quantile: float = 99.0):
        super().__init__(n_components=n_components)
        self.ref_quantile = ref_quantile
        self.t2_ref_: Optional[float] = None
        self.spe_ref_: Optional[float] = None

    def _stats(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = self._standardize(np.asarray(X, dtype=np.float64))
        z = self.pca.transform(Xs)
        var = np.maximum(self.pca.explained_variance_, 1e-12)
        t2 = (z**2 / var).sum(axis=1)
        spe = ((Xs - self.pca.inverse_transform(z)) ** 2).sum(axis=1)
        return t2, spe

    def _update_refs(self, X: np.ndarray) -> None:
        t2, spe = self._stats(X)
        t2_ref, spe_ref = np.percentile(t2, self.ref_quantile), np.percentile(spe, self.ref_quantile)
        if self.t2_ref_ is None:
            self.t2_ref_, self.spe_ref_ = float(t2_ref), float(spe_ref)
        else:  # one odd month must not redefine the scale outright
            self.t2_ref_ = float(0.5 * self.t2_ref_ + 0.5 * t2_ref)
            self.spe_ref_ = float(0.5 * self.spe_ref_ + 0.5 * spe_ref)

    def fit(self, X: np.ndarray) -> "HotellingPCA":
        super().fit(X)
        self.t2_ref_ = self.spe_ref_ = None
        self._update_refs(X)
        return self

    def partial_fit(self, X: np.ndarray) -> "HotellingPCA":
        super().partial_fit(X)
        self._update_refs(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.t2_ref_ is None:
            raise RuntimeError("HotellingPCA must be fit before scoring")
        t2, spe = self._stats(X)
        return np.maximum(t2 / max(self.t2_ref_, 1e-12), spe / max(self.spe_ref_, 1e-12))
