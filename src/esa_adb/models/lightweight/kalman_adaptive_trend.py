"""Adaptive-noise extension of KalmanLocalTrend.

The plain filter sets its noise levels once, from the first fit batch, and never
updates them. A channel whose real noise level drifts over the mission ends up
scored against a stale estimate.

This version updates the observation noise R with a simple exponential average,
but only from rows that already look normal (standardized innovation under a
gate). A real anomaly's own large innovation never gets to raise R, so it can't
teach the filter to treat its own size as normal. Process noise Q is left fixed:
an ungated joint update of both is known to be unstable (Sarkka & Hartikainen
2013), so this only touches the safer half of that idea.
"""
from __future__ import annotations

import numpy as np

from esa_adb.models.lightweight.kalman_local_trend import KalmanLocalTrend


class KalmanAdaptiveTrend(KalmanLocalTrend):
    def __init__(self, adapt_rate: float = 0.01, gate: float = 3.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.adapt_rate = adapt_rate
        self.gate = gate

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("KalmanAdaptiveTrend must be fit before scoring")
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

            k_level = pp00 / s
            k_trend = pp01 / s
            self.state[:, 0] = level_pred + k_level * innovation
            self.state[:, 1] = trend_pred + k_trend * innovation
            self.cov[:, 0, 0] = (1 - k_level) * pp00
            self.cov[:, 0, 1] = (1 - k_level) * pp01
            self.cov[:, 1, 0] = self.cov[:, 0, 1]
            self.cov[:, 1, 1] = pp11 - k_trend * pp01

            # Only let normal-looking rows inform the noise estimate, so a real
            # anomaly can't teach the filter to treat its own size as normal.
            normal = np.abs(std_innovation) <= self.gate
            observed_r = (innovation**2 - pp00)
            self.R[normal] = np.maximum(
                (1 - self.adapt_rate) * self.R[normal] + self.adapt_rate * np.maximum(observed_r[normal], 1e-12),
                1e-12,
            )
        return out.max(axis=1)
