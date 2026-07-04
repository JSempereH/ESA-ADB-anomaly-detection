from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from esa_adb.preprocessing.windowing import WindowSpec, window_row_slices


class AnomalyDetector(ABC):
    """Common interface for every detector in `models/` (lightweight and heavy alike).

    `fit()` is only ever expected to see nominal data -- every detector here is
    semi-supervised/unsupervised by design, matching the production constraint of not
    having curated anomaly labels for a newly launched satellite. `score()` returns
    higher-is-more-anomalous values so detectors are directly comparable regardless of
    their internal convention (e.g. scikit-learn's `IsolationForest` uses the opposite
    sign internally -- wrappers here normalize it away).
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "AnomalyDetector": ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.score(X) > threshold


def build_window_feature_matrix(values: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Flatten each sliding window of `values` (n_rows, n_channels) into one feature row.

    Used by window-based detectors (e.g. Windowed Isolation Forest) that need a single
    fixed-size vector per window rather than a per-timestep score.
    """
    n_channels = values.shape[1] if values.ndim > 1 else 1
    rows = [values[start:end].reshape(-1) for _, start, end in window_row_slices(values.shape[0], spec)]
    if not rows:
        return np.empty((0, spec.length * n_channels))
    return np.stack(rows)


def window_scores_to_row_scores(window_scores: np.ndarray, n_rows: int, spec: WindowSpec) -> np.ndarray:
    """Broadcast per-window scores back onto per-row scores (max over covering windows).

    A row covered by several overlapping windows takes the highest (most anomalous) of
    their scores -- consistent with `label_windows`' "any covering window flags this
    point" semantics for ground truth.
    """
    row_scores = np.full(n_rows, -np.inf)
    for window_index, start, end in window_row_slices(n_rows, spec):
        row_scores[start:end] = np.maximum(row_scores[start:end], window_scores[window_index])
    return row_scores


def broadcast_window_pred_to_rows(window_pred: np.ndarray, n_rows: int, spec: WindowSpec) -> np.ndarray:
    """Broadcast per-window bool predictions onto per-row bool predictions (OR over
    covering windows) -- the boolean analogue of `window_scores_to_row_scores`. Needed to
    compare a window-based detector's predictions (e.g. `WindowedIsolationForest`, whose
    `predict()` uses scikit-learn's own contamination-based decision at the window level)
    against row-indexed ground truth like `evaluation.evaluate_events`.
    """
    row_pred = np.zeros(n_rows, dtype=bool)
    window_pred = np.asarray(window_pred, dtype=bool)
    for window_index, start, end in window_row_slices(n_rows, spec):
        if window_pred[window_index]:
            row_pred[start:end] = True
    return row_pred


def channel_matrix(df: pl.DataFrame, channels: list[str]) -> np.ndarray:
    """Extract a (n_rows, n_channels) float64 array of the given columns from a Polars DataFrame."""
    return df.select(channels).to_numpy().astype(np.float64)
