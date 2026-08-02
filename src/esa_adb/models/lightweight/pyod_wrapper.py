from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector


class PyODDetector(AnomalyDetector):
    """Generic adapter for any `pyod.models.*` detector, parameterized by an
    `estimator_factory` (zero-arg callable returning a fresh, unfitted PyOD estimator)
    rather than one class per algorithm -- ECOD/COPOD/IForest/etc. all share this same
    fit/score contract, so one wrapper class covers all of them.

    Almost every PyOD detector only exposes batch `fit()`/`decision_function()` -- no
    `partial_fit`. `partial_fit` here fakes online behavior the same way this project's
    `AutoencoderRunner` (replay buffer) and `NearestNominalWindow` (reference window
    set) already do: keep a reservoir-sampled, size-bounded buffer of every fit row
    ever seen, and re-fit a BRAND NEW estimator from scratch on that buffer each call.
    Every row keeps an equal probability of being retained regardless of when it
    arrived, so the buffer stays representative of the whole mission history rather
    than just the latest fold -- exactly the same algorithm as
    `NearestNominalWindow._add_windows`/`AutoencoderRunner._update_replay`, applied to
    raw rows instead of windows.

    `decision_function` already follows PyOD's own "higher = more abnormal" convention
    for every detector in the library, matching `AnomalyDetector.score`'s contract
    directly -- no sign flip needed (unlike scikit-learn's raw `IsolationForest`).
    """

    def __init__(self, estimator_factory: Callable[[], object], buffer_rows: int = 50_000, seed: int = 42):
        self.estimator_factory = estimator_factory
        self.buffer_rows = buffer_rows
        self._rng = np.random.default_rng(seed)
        self._fitted = None
        self._buffer: Optional[np.ndarray] = None
        self._buffer_seen = 0

    def fit(self, X: np.ndarray) -> "PyODDetector":
        self._fitted = self.estimator_factory().fit(X)
        return self

    def _reservoir_update(self, X: np.ndarray) -> None:
        cap = self.buffer_rows
        if self._buffer is None:
            self._buffer = X[:cap].copy()
            self._buffer_seen = self._buffer.shape[0]
            X = X[cap:]
        for row in X:
            self._buffer_seen += 1
            if self._buffer.shape[0] < cap:
                self._buffer = np.vstack([self._buffer, row])
            else:
                j = self._rng.integers(0, self._buffer_seen)
                if j < cap:
                    self._buffer[j] = row

    def partial_fit(self, X: np.ndarray) -> "PyODDetector":
        self._reservoir_update(X)
        self._fitted = self.estimator_factory().fit(self._buffer)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("PyODDetector must be fit before scoring")
        return np.asarray(self._fitted.decision_function(X))
