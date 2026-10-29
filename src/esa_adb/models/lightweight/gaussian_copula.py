"""Gaussian-copula joint anomaly score.

ECOD and COPOD score each channel's tail probability on its own and then just add
the scores up, treating channels as independent. This detector keeps the same
empirical-CDF idea but scores channels jointly instead.

For each channel, the reference sample gives an empirical CDF. A new value's rank
in that reference is its probability-integral transform, a number between 0 and 1,
which maps to a standard normal quantile. Under normal conditions every channel's
quantile is roughly standard normal, no matter the channel's original units, which
is also what makes ECOD's per-channel scores comparable in the first place.

Instead of adding up each channel's quantile on its own, we score the whole row at
once with a Mahalanobis distance against a fitted correlation matrix. A row where
no single channel looks unusual, but the channels no longer move together the way
they normally do, now scores high too.

The correlation matrix is estimated with Spearman rank correlation, which limits
how much a single bad row can pull it around, plus a MAD-based filter that drops
the most extreme rows before fitting, the same filter used by the Kalman variants
in this project.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm, rankdata

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.robust import mad_filter

_EPS = 1e-6


class GaussianCopulaDetector(AnomalyDetector):
    def __init__(self, corr_reg: float = 0.1, max_reference_rows: int = 20_000, seed: int = 42) -> None:
        self.corr_reg = corr_reg
        self.max_reference_rows = max_reference_rows
        self.seed = seed
        self.reference_: Optional[np.ndarray] = None  # sorted per-channel reference values, (r, d)
        self.corr_inv_: Optional[np.ndarray] = None  # (d, d)
        self.d_: Optional[int] = None

    def _pit_u(self, X: np.ndarray) -> np.ndarray:
        """Per-channel rank transform against the frozen reference, in (0, 1).

        Subclasses that target a different marginal (TCopulaDetector) only
        override `_pit_z`, never this method, so they share the same filtering
        and correlation matrix and differ only in the final scoring formula."""
        r = self.reference_.shape[0]
        u = np.empty_like(X, dtype=np.float64)
        for c in range(X.shape[1]):
            ranks = np.searchsorted(self.reference_[:, c], X[:, c], side="right")
            u[:, c] = np.clip(ranks / r, _EPS, 1 - _EPS)
        return u

    def _pit_z(self, X: np.ndarray) -> np.ndarray:
        """Maps the rank transform to standard normal quantiles."""
        return norm.ppf(self._pit_u(X))

    def fit(self, X: np.ndarray) -> "GaussianCopulaDetector":
        X = np.asarray(X, dtype=np.float64)
        self.d_ = X.shape[1]
        ref = X
        if ref.shape[0] > self.max_reference_rows:
            idx = np.random.default_rng(self.seed).choice(ref.shape[0], size=self.max_reference_rows, replace=False)
            ref = ref[idx]
        self.reference_ = np.sort(ref, axis=0)

        u = self._pit_u(X)
        keep = mad_filter(u, k=3.5, max_reject_fraction=0.1)
        clean = u[keep] if keep.sum() >= max(2 * self.d_, 30) else u
        ranks = np.apply_along_axis(rankdata, 0, clean)
        corr = np.corrcoef(ranks, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = (1 - self.corr_reg) * corr + self.corr_reg * np.eye(self.d_)
        self.corr_inv_ = np.linalg.inv(corr)
        return self

    def partial_fit(self, X: np.ndarray) -> "GaussianCopulaDetector":
        # There is no incremental update for an empirical-CDF reference or a rank
        # correlation matrix, both need the full batch to define ranks. Refit fresh
        # each fold instead.
        return self.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("GaussianCopulaDetector must be fit before scoring")
        z = self._pit_z(np.asarray(X, dtype=np.float64))
        d2 = np.einsum("ti,ij,tj->t", z, self.corr_inv_, z)
        return (d2 - self.d_) / np.sqrt(2.0 * self.d_)
