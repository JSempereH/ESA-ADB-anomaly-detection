from __future__ import annotations

from typing import Optional

import numpy as np
import stumpy

from esa_adb.models.base import AnomalyDetector


class MatrixProfileDetector(AnomalyDetector):
    """Per-channel Matrix Profile (Yeh et al., ICDM 2016/2017) via stumpy's AB-join.

    For each channel, every subsequence of length m in the scored window is
    compared against its nearest neighbor in a reference window. The nearest-
    neighbor distance is the anomaly score: a large distance means this local
    shape has no close match anywhere in the reference (a "discord"). Per-channel
    scores are summed by default, or combined with max.

    No training, no gradient descent, just shape matching. NASA JPL used the same
    per-channel-sum recipe on Cassini telemetry as an Europa Clipper testbed
    (Daigavane & Wagstaff, MileTS'20) and it was their best-performing method.

    Two compute-driven shortcuts:
    - The reference window is the last ref_rows rows of the fit batch, not the
      whole month. A full month on both sides would take minutes per channel.
      ref_rows=5000 keeps each fold under a minute.
    - There is no incremental multivariate Matrix Profile algorithm, so fit()
      and partial_fit() both just replace the reference with the latest batch.
    """

    def __init__(self, m: int = 50, ref_rows: int = 5_000, fusion: str = "sum"):
        self.m = m
        self.ref_rows = ref_rows
        self.fusion = fusion
        self.reference_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MatrixProfileDetector":
        X = np.asarray(X, dtype=np.float64)
        self.reference_ = X[-self.ref_rows :]
        return self

    def partial_fit(self, X: np.ndarray) -> "MatrixProfileDetector":
        return self.fit(X)

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("MatrixProfileDetector must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        n_rows, n_channels = X.shape
        out = np.zeros((n_rows, n_channels))
        valid = n_rows - self.m + 1
        if valid <= 0 or self.reference_.shape[0] <= self.m:
            return out
        for c in range(n_channels):
            query = X[:, c]
            ref = self.reference_[:, c]
            nn_dist = stumpy.stump(query, m=self.m, T_B=ref, ignore_trivial=False)[:, 0].astype(np.float64)
            out[: len(nn_dist), c] = nn_dist
            out[len(nn_dist) :, c] = nn_dist[-1] if len(nn_dist) else 0.0
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        per_channel = self.score_per_channel(X)
        return per_channel.sum(axis=1) if self.fusion == "sum" else per_channel.max(axis=1)

    def most_anomalous_channels(self, X: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        z = self.score_per_channel(X)
        top_n = min(top_n, z.shape[1])
        order = np.argsort(-z, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]
