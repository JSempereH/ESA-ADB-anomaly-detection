"""ADDAEIL (Li, Nair & Wang, 2025, Algorithms 18(6):359): a drift-aware isolation
forest ensemble. Unlike an online isolation forest that grows and updates a single
tree structure, this one keeps a fixed-size forest of ordinary trees and replaces
whichever trees a drift test flags, each fold, instead of mutating any tree in
place.

Two steps each fold:
1. Compare the current window's score distribution against a base window with
   a Kolmogorov-Smirnov test. The statistic (already between 0 and 1) sets how
   large a fraction of the forest to replace, capped at replace_fraction_cap.
2. For each tree, compare its own path-length distribution on the base vs.
   current window with a Mann-Whitney U test. The trees with the lowest
   p-values, the ones most affected by drift, get replaced with fresh trees.

The paper's own experiments are univariate, so it does not say how to combine
channels. Here both steps just use the forest's own multivariate path lengths
directly, since each tree already splits on a mix of channels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.ensemble import IsolationForest

from esa_adb.models.base import AnomalyDetector


def _c_factor(n: float) -> float:
    """Average path length of an unsuccessful search in a binary search tree over
    n points. This is the standard Isolation Forest normalizing constant."""
    n = max(n, 2.0)
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class ADDAEILForest(AnomalyDetector):
    """Scores are computed by hand from each tree's decision path instead of
    sklearn's IsolationForest.score_samples. sklearn caches per-tree bookkeeping
    from the original fit, and a replacement tree usually has a different number
    of leaves, which crashes that cached scoring path once any tree is swapped
    in. Reading path length straight from each tree avoids that."""

    def __init__(
        self,
        n_estimators: int = 100,
        replace_fraction_cap: float = 0.6,
        max_ref_rows: int = 2_000,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.replace_fraction_cap = replace_fraction_cap
        self.max_ref_rows = max_ref_rows
        self.seed = seed
        self.model: Optional[IsolationForest] = None
        self.base_ref_: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

    def _subsample_ref(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] <= self.max_ref_rows:
            return X
        idx = self._rng.choice(X.shape[0], size=self.max_ref_rows, replace=False)
        return X[idx]

    def fit(self, X: np.ndarray) -> "ADDAEILForest":
        X = np.asarray(X, dtype=np.float64)
        self.model = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(X)
        self.base_ref_ = self._subsample_ref(X)
        return self

    def partial_fit(self, X: np.ndarray) -> "ADDAEILForest":
        if self.model is None:
            return self.fit(X)
        X = np.asarray(X, dtype=np.float64)
        cur_ref = self._subsample_ref(X)

        # Module 1: drift intensity from the overall score-distribution shift.
        base_scores = self.score(self.base_ref_)
        cur_scores = self.score(cur_ref)
        drift_intensity = ks_2samp(base_scores, cur_scores).statistic  # already in [0, 1]
        replace_ratio = min(self.replace_fraction_cap, drift_intensity)
        n_replace = int(round(replace_ratio * self.n_estimators))
        if n_replace == 0:
            self.base_ref_ = cur_ref
            return self

        # Module 2: rank trees by how significantly THEIR OWN path-length
        # distribution shifted (lowest p-value = most drift-affected).
        p_values = np.empty(self.n_estimators)
        for i, tree in enumerate(self.model.estimators_):
            base_path = np.asarray(tree.decision_path(self.base_ref_).sum(axis=1)).ravel()
            cur_path = np.asarray(tree.decision_path(cur_ref).sum(axis=1)).ravel()
            p_values[i] = mannwhitneyu(base_path, cur_path, alternative="two-sided").pvalue

        replace_idx = np.argsort(p_values)[:n_replace]
        # max_samples must match the original forest's value, or the replacement
        # trees end up with a different leaf count and the scoring below breaks.
        fresh = IsolationForest(
            n_estimators=n_replace, max_samples=self.model.max_samples_, random_state=self._rng.integers(1_000_000)
        ).fit(X)
        for slot, new_tree, new_features in zip(replace_idx, fresh.estimators_, fresh.estimators_features_):
            self.model.estimators_[slot] = new_tree
            self.model.estimators_features_[slot] = new_features

        self.base_ref_ = cur_ref
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ADDAEILForest must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        path_lengths = np.zeros(X.shape[0])
        for tree in self.model.estimators_:
            path_lengths += np.asarray(tree.decision_path(X).sum(axis=1)).ravel()
        avg_path_length = path_lengths / len(self.model.estimators_)
        c = _c_factor(self.model.max_samples_)
        # Shorter average path means easier to isolate, so the score below is
        # already higher for more anomalous rows, no sign flip needed.
        return 2.0 ** (-avg_path_length / c)
