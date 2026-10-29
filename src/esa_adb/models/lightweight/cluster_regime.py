"""Replication attempt of the Mission-2 "clustering-based framework" from
arXiv:2609.10017 (Alcarria, Sanchez, Sempere et al., SPAICE 2026, Section 2.5.2): robust
scaling + MAD outlier filtering -> PCA to 95% variance -> k-means (k chosen by the Gap
Statistic, Tibshirani et al. 2001) over operational regimes -> Local Outlier Factor for
local density deviation -> combined score = 0.7*LOF + 0.3*distance-to-nearest-centroid.

This is a replication ATTEMPT, not a verified port: the paper gives the pipeline stages
and the 0.7/0.3 weighting explicitly, but not every implementation detail (e.g. the
exact MAD-filtering cutoff, the Gap Statistic's own search range/bootstrap count, or how
the two sub-scores are put on a comparable scale before the weighted sum). Those are
filled in here with documented, ordinary choices rather than left unstated. `fit_subsample`
is one such addition the paper doesn't need to address: a real monthly fold at this
project's scale is 10^5+ rows, where the Gap Statistic's repeated k-means fits and LOF's
neighbor search would otherwise dominate the whole pipeline's runtime -- subsampling
before fitting (never before scoring) is the standard way to keep this usable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

from esa_adb.models.base import AnomalyDetector


def _mad_filter(X: np.ndarray, k: float = 3.5, max_reject_fraction: float = 0.1) -> np.ndarray:
    """Row-wise robust outlier mask (True = keep), matching the paper's "robust
    sample-level filtering based on MAD" preprocessing step. A row's deviation is its
    mean absolute per-channel z-score (robust median/MAD z-score), rejected if that
    exceeds `k`, capped at `max_reject_fraction` of the batch to avoid filtering away
    most of a fold's data if the whole fold looks unusual."""
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad[mad == 0] = 1.0
    z = np.abs(X - median) / (1.4826 * mad)
    deviation = z.mean(axis=1)
    threshold = np.median(deviation) + k * (np.median(np.abs(deviation - np.median(deviation))) * 1.4826 + 1e-9)
    reject = deviation > threshold
    max_reject = int(max_reject_fraction * X.shape[0])
    if reject.sum() > max_reject:
        order = np.argsort(-deviation, kind="stable")
        reject = np.zeros(X.shape[0], dtype=bool)
        reject[order[:max_reject]] = True
    return ~reject


def _gap_statistic(X: np.ndarray, k_range: range, b: int = 5, seed: int = 42) -> int:
    """Tibshirani et al. (2001): compare k-means' within-cluster dispersion (its own
    `inertia_`, equal up to a constant to the paper's pairwise-distance formulation)
    against `b` uniform-random reference datasets of the same bounding box, picking the
    smallest k whose gap is within one standard error of the next candidate's."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    ks = list(k_range)
    gaps, sks = [], []
    for k in ks:
        wk = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X).inertia_
        ref_logs = []
        for _ in range(b):
            ref = rng.uniform(lo, hi, size=X.shape)
            ref_logs.append(np.log(max(KMeans(n_clusters=k, n_init=3, random_state=seed).fit(ref).inertia_, 1e-12)))
        gaps.append(np.mean(ref_logs) - np.log(max(wk, 1e-12)))
        sks.append(np.std(ref_logs) * np.sqrt(1 + 1.0 / b))
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return ks[i]
    return ks[-1]


class ClusterRegimeDetector(AnomalyDetector):
    def __init__(
        self,
        variance_threshold: float = 0.95,
        k_range: range = range(2, 9),
        gap_bootstrap: int = 5,
        lof_neighbors: int = 20,
        lof_weight: float = 0.7,
        centroid_weight: float = 0.3,
        mad_k: float = 3.5,
        fit_subsample: int = 8000,
        seed: int = 42,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.k_range = k_range
        self.gap_bootstrap = gap_bootstrap
        self.lof_neighbors = lof_neighbors
        self.lof_weight = lof_weight
        self.centroid_weight = centroid_weight
        self.mad_k = mad_k
        self.fit_subsample = fit_subsample
        self.seed = seed
        self.median_: Optional[np.ndarray] = None
        self.mad_: Optional[np.ndarray] = None
        self.pca_: Optional[PCA] = None
        self.kmeans_: Optional[KMeans] = None
        self.lof_: Optional[LocalOutlierFactor] = None
        self._lof_mean = 0.0
        self._lof_std = 1.0
        self._cd_mean = 0.0
        self._cd_std = 1.0

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median_) / self.mad_

    def fit(self, X: np.ndarray) -> "ClusterRegimeDetector":
        X = np.asarray(X, dtype=np.float64)
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0) * 1.4826
        mad[mad == 0] = 1.0
        self.mad_ = mad

        scaled = self._scale(X)
        keep = _mad_filter(scaled, k=self.mad_k)
        clean = scaled[keep]
        if clean.shape[0] < max(self.k_range.stop, self.lof_neighbors + 1):
            clean = scaled  # too few rows survive filtering (e.g. a short fold); don't starve the fit

        # The paper doesn't address runtime at this scale; a full mission-month (10^5+
        # rows) makes the Gap Statistic's repeated KMeans fits and LOF's O(n log n)
        # neighbor search the dominant cost of the whole pipeline by a wide margin.
        # Subsampling before PCA/k-means/LOF (but scoring every row against the fitted
        # model afterward) is the standard, documented way to keep a regime-clustering
        # pipeline usable at this project's walk-forward scale.
        if clean.shape[0] > self.fit_subsample:
            idx = np.random.default_rng(self.seed).choice(clean.shape[0], size=self.fit_subsample, replace=False)
            clean = clean[idx]

        n_components = min(clean.shape[1], max(1, clean.shape[0] - 1))
        self.pca_ = PCA(n_components=n_components, random_state=self.seed).fit(clean)
        reduced = self.pca_.transform(clean)
        cumvar = np.cumsum(self.pca_.explained_variance_ratio_)
        n_keep = max(1, int(np.searchsorted(cumvar, self.variance_threshold) + 1))
        reduced = reduced[:, :n_keep]
        self._n_keep = n_keep

        k_range = range(self.k_range.start, min(self.k_range.stop, max(self.k_range.start + 1, reduced.shape[0] // 10)))
        if k_range.stop <= k_range.start:
            k_range = range(self.k_range.start, self.k_range.start + 1)
        k = _gap_statistic(reduced, k_range, b=self.gap_bootstrap, seed=self.seed)
        self.kmeans_ = KMeans(n_clusters=k, n_init=5, random_state=self.seed).fit(reduced)

        self.lof_ = LocalOutlierFactor(n_neighbors=min(self.lof_neighbors, reduced.shape[0] - 1), novelty=True)
        self.lof_.fit(reduced)

        lof_raw = -self.lof_.score_samples(reduced)
        cd_raw = self._centroid_distance(reduced)
        self._lof_mean, self._lof_std = lof_raw.mean(), max(lof_raw.std(), 1e-8)
        self._cd_mean, self._cd_std = cd_raw.mean(), max(cd_raw.std(), 1e-8)
        return self

    def _centroid_distance(self, reduced: np.ndarray) -> np.ndarray:
        centers = self.kmeans_.cluster_centers_
        d2 = ((reduced[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.sqrt(d2.min(axis=1))

    def _reduce(self, X: np.ndarray) -> np.ndarray:
        scaled = self._scale(X)
        return self.pca_.transform(scaled)[:, : self._n_keep]

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("ClusterRegimeDetector must be fit before scoring")
        reduced = self._reduce(np.asarray(X, dtype=np.float64))
        lof = (-self.lof_.score_samples(reduced) - self._lof_mean) / self._lof_std
        cd = (self._centroid_distance(reduced) - self._cd_mean) / self._cd_std
        return self.lof_weight * lof + self.centroid_weight * cd
