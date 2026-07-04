from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from esa_adb.models.base import AnomalyDetector, build_window_feature_matrix
from esa_adb.preprocessing.windowing import WindowSpec

FeatureExtractor = Callable[[np.ndarray, WindowSpec], np.ndarray]


class WindowedIsolationForest(AnomalyDetector):
    """Isolation Forest over sliding windows.

    The ESA-ADB paper's own best-performing unsupervised algorithm for Mission2
    (corrected event-wise F0.5 = 0.949) -- competitive with, and sometimes better than,
    the deep-learning baselines it was benchmarked against. By default each window's
    channel values across `spec.length` samples are flattened into one feature vector
    (the paper used a window length of 17, down from a default of 100, specifically to
    bound memory); pass `feature_extractor=models.window_features.extract_window_features`
    for a statistical+spectral feature vector per window instead of raw values.

    `fit()` expects nominal-only rows (e.g. a mission's train split with
    `preprocessing.labeling.filter_nominal_rows` applied).
    """

    def __init__(self, spec: WindowSpec, feature_extractor: FeatureExtractor = build_window_feature_matrix, **iforest_kwargs):
        self.spec = spec
        self.feature_extractor = feature_extractor
        iforest_kwargs.setdefault("random_state", 42)
        self.model = IsolationForest(**iforest_kwargs)

    def fit(self, values: np.ndarray) -> "WindowedIsolationForest":
        X = self.feature_extractor(values, self.spec)
        self.model.fit(X)
        return self

    def partial_fit(self, values: np.ndarray, n_new_estimators: int = 100) -> "WindowedIsolationForest":
        """Add `n_new_estimators` MORE trees fit on this new batch, on top of whatever
        trees already exist, via scikit-learn's `warm_start=True` -- `IsolationForest`
        has no native `partial_fit`, but `warm_start` is its documented way to grow a
        forest without discarding earlier trees: calling `.fit()` again with a bigger
        `n_estimators` fits only the newly added trees on the batch passed this time,
        while the existing trees (still reflecting whatever batch built them) are kept
        as-is. This is what "update from the new month, don't forget the rest" means
        for a tree ensemble, in contrast to `fit()`'s plain batch retrain.
        """
        X = self.feature_extractor(values, self.spec)
        self.model.set_params(warm_start=True, n_estimators=self.model.n_estimators + n_new_estimators)
        self.model.fit(X)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        X = self.feature_extractor(values, self.spec)
        # sklearn's score_samples: lower = more anomalous. Flip sign so this class's
        # convention (higher = more anomalous) matches every other detector in models/.
        return -self.model.score_samples(X)

    def predict(self, values: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        if threshold is not None:
            return self.score(values) > threshold
        X = self.feature_extractor(values, self.spec)
        return self.model.predict(X) == -1  # sklearn's own contamination-based decision
