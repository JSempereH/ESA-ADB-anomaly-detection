"""Robust (Huber-type) extension of KalmanLocalTrend.

A plain Kalman filter assumes roughly Gaussian noise, so a genuine anomaly can
drag the state estimate toward itself through the Kalman gain. The next
observation then gets compared against a state that the anomaly already pulled
off course, which understates how unusual it really is.

This version clips the innovation before it updates the state, so one bad
reading can't drag the state as far, but still reports the raw, unclipped
innovation as the score. A genuine anomaly is flagged at full strength; only
its pull on the next fold's reference state is limited. Follows the general
Huber-robust Kalman recipe (Ruckdeschel, Spangl & Pupashenko 2014).
"""
from __future__ import annotations

import numpy as np

from esa_adb.models.lightweight.kalman_local_trend import KalmanLocalTrend


class KalmanRobustTrend(KalmanLocalTrend):
    def __init__(self, huber_c: float = 1.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.huber_c = huber_c

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("KalmanRobustTrend must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        out = np.empty((n, d))
        for t in range(n):
            level_pred = self.state[:, 0] + self.state[:, 1]
            trend_pred = self.state[:, 1]
            p00, p01, p11 = self.cov[:, 0, 0], self.cov[:, 0, 1], self.cov[:, 1, 1]
            pp00 = p00 + 2 * p01 + p11 + self.Q[:, 0, 0]
            pp01 = p01 + p11
            pp11 = p11 + self.Q[:, 1, 1]

            innovation = X[t] - level_pred
            s = pp00 + self.R
            std_innovation = innovation / np.sqrt(np.maximum(s, 1e-12))
            out[t] = np.abs(std_innovation)

            # Clip only the copy used to update the state. The score above stays
            # at the raw magnitude, so a real anomaly is never under-reported.
            clipped_std = np.clip(std_innovation, -self.huber_c, self.huber_c)
            clipped_innovation = clipped_std * np.sqrt(np.maximum(s, 1e-12))

            k_level = pp00 / s
            k_trend = pp01 / s
            self.state[:, 0] = level_pred + k_level * clipped_innovation
            self.state[:, 1] = trend_pred + k_trend * clipped_innovation
            self.cov[:, 0, 0] = (1 - k_level) * pp00
            self.cov[:, 0, 1] = (1 - k_level) * pp01
            self.cov[:, 1, 0] = self.cov[:, 0, 1]
            self.cov[:, 1, 1] = pp11 - k_trend * pp01
        return out.max(axis=1)
