from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from esa_adb.models.base import AnomalyDetector, build_window_feature_matrix
from esa_adb.preprocessing.windowing import WindowSpec


class NearestNominalWindow(AnomalyDetector):
    """Distance-to-nearest-nominal-window detector -- matrix-profile-inspired
    (DAMP-style "how far is this subsequence from anything nominal ever seen"), adapted
    to this project's constraints: multivariate (a window spans ALL channels jointly,
    where a true matrix profile is univariate), tumbling windows on the preprocessed
    grid (the same `WindowSpec` convention as `WindowedIsolationForest`), and a BOUNDED
    reservoir-sampled reference set so memory never grows with mission length.

    The appeal for a "few alarms, well placed" objective: unlike density/tree methods,
    this can memorize *rare but nominal* patterns -- a maneuver that happened twice in
    training keeps a nearby reference window and scores low forever after, exactly the
    property the ESA-ADB paper cites when calling matrix-profile methods promising.
    `fit`/`partial_fit` only ever see nominal rows (project convention), so every
    reference window is nominal by construction. Score per window = euclidean distance
    to the nearest reference window / sqrt(window_dim), broadcast back to rows.
    """

    def __init__(self, spec: WindowSpec, max_reference_windows: int = 8000, seed: int = 42):
        self.spec = spec
        self.max_reference_windows = max_reference_windows
        self._rng = np.random.default_rng(seed)
        self._reference: Optional[np.ndarray] = None
        self._windows_seen = 0
        self._nn: Optional[NearestNeighbors] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        """Per-channel z-scoring with stats fixed from the FIRST batch ever seen (the
        `Autoencoder.partial_fit` convention: don't silently redefine "normal" later).
        Without this, euclidean window distance is dominated by whichever channels have
        the largest raw units, making most channels invisible to the detector.
        """
        if self.mean_ is None:
            self.mean_ = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            self.std_ = std
        return (values - self.mean_) / self.std_

    def _add_windows(self, windows: np.ndarray) -> None:
        """Reservoir sampling: every window ever seen keeps an equal probability of
        being in the bounded reference set, so it stays representative of the whole
        mission, not just the latest batch."""
        cap = self.max_reference_windows
        for w in windows:
            self._windows_seen += 1
            if self._reference is None:
                self._reference = w[np.newaxis, :].copy()
            elif self._reference.shape[0] < cap:
                self._reference = np.vstack([self._reference, w])
            else:
                j = self._rng.integers(0, self._windows_seen)
                if j < cap:
                    self._reference[j] = w
        self._nn = None  # invalidate the index; rebuilt lazily on next score()

    def fit(self, values: np.ndarray) -> "NearestNominalWindow":
        self._reference = None
        self._windows_seen = 0
        self._nn = None
        self.mean_ = None
        self.std_ = None
        self._add_windows(build_window_feature_matrix(self._standardize(values), self.spec))
        return self

    def partial_fit(self, values: np.ndarray) -> "NearestNominalWindow":
        self._add_windows(build_window_feature_matrix(self._standardize(values), self.spec))
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        """Per-WINDOW nearest-neighbor distance (higher = less like anything nominal),
        in the same window-level convention as `WindowedIsolationForest.score`."""
        if self._reference is None:
            raise RuntimeError("NearestNominalWindow must be fit before scoring")
        windows = build_window_feature_matrix(self._standardize(values), self.spec)
        if windows.shape[0] == 0:
            return np.empty(0)
        if self._nn is None:
            self._nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(self._reference)
        distances, _ = self._nn.kneighbors(windows)
        return distances[:, 0] / np.sqrt(windows.shape[1])

    def predict(self, values: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        return self.score(values) > threshold
