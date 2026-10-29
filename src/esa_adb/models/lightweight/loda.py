"""Loda (Pevny, Machine Learning 2016): an ensemble of one-dimensional histograms over
sparse random projections. Score = mean negative log-probability of a row's projections.

Chosen because it is natively streaming (counts only ever accumulate, nothing is
forgotten or re-trained), costs O(projections x nnz) per row, and has no learned
parameters a contaminated fit batch could bend: contamination just adds a little mass
to the histogram tails. Bin edges are fixed from the first batch's 0.1-99.9 percentiles
and the outermost bins are open-ended, so an extreme projection lands in a sparsely
populated tail bin instead of being clipped to a typical one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector


class LodaDetector(AnomalyDetector):
    def __init__(self, n_projections: int = 200, n_bins: int = 64, seed: int = 42):
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.seed = seed
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.proj_: Optional[np.ndarray] = None  # (d, k)
        self.edges_: Optional[np.ndarray] = None  # (k, n_bins - 1) inner edges
        self.counts_: Optional[np.ndarray] = None  # (k, n_bins)

    def _project(self, X: np.ndarray) -> np.ndarray:
        Xs = np.clip((np.asarray(X, dtype=np.float64) - self.mean_) / self.std_, -10.0, 10.0)
        return Xs @ self.proj_

    def _bin(self, P: np.ndarray) -> np.ndarray:
        return np.stack([np.searchsorted(self.edges_[j], P[:, j]) for j in range(P.shape[1])], axis=1)

    def _init(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        nnz = max(1, int(round(np.sqrt(d))))
        proj = np.zeros((d, self.n_projections))
        for j in range(self.n_projections):
            proj[rng.choice(d, size=nnz, replace=False), j] = rng.normal(size=nnz)
        self.proj_ = proj
        P = self._project(X)
        lo, hi = np.percentile(P, 0.1, axis=0), np.percentile(P, 99.9, axis=0)
        hi = np.where(hi > lo, hi, lo + 1.0)
        self.edges_ = np.stack([np.linspace(lo[j], hi[j], self.n_bins - 1) for j in range(self.n_projections)])
        self.counts_ = np.zeros((self.n_projections, self.n_bins))

    def partial_fit(self, X: np.ndarray) -> "LodaDetector":
        X = np.asarray(X, dtype=np.float64)
        if self.proj_ is None:
            self._init(X)
        idx = self._bin(self._project(X))
        for j in range(self.n_projections):
            self.counts_[j] += np.bincount(idx[:, j], minlength=self.n_bins)
        return self

    def fit(self, X: np.ndarray) -> "LodaDetector":
        self.proj_ = None
        return self.partial_fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.proj_ is None:
            raise RuntimeError("LodaDetector must be fit before scoring")
        idx = self._bin(self._project(X))
        prob = (self.counts_ + 1.0) / (self.counts_.sum(axis=1, keepdims=True) + self.n_bins)
        return -np.log(prob[np.arange(self.n_projections)[None, :], idx]).mean(axis=1)
