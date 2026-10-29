"""Per-channel local-level/local-trend Kalman filter.

Each channel gets its own [level, trend] state, updated with the standard
local-linear-trend model:
    level_t = level_{t-1} + trend_{t-1} + noise
    trend_t = trend_{t-1} + noise
    y_t     = level_t + observation noise
All channels run independently but are vectorized together instead of looped
in Python.

The score for a row is its worst channel's standardized innovation: how far
the observed value is from the predicted level, divided by the predicted
uncertainty. This adapts to each channel's own noise and trend online, instead
of comparing against a fixed threshold.

Q and R (process and observation noise) are set once from the first fit
batch's variance, not fit by maximum likelihood. R is a fraction of the
channel's own variance; Q is a smaller fraction of how much it changes step to
step. There is no separate "freeze after fitting" step: every new observation
updates the running state, so calling score() on calibration data and then on
eval data is the normal, intended way to use this.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector


class KalmanLocalTrend(AnomalyDetector):
    def __init__(
        self,
        obs_noise_fraction: float = 0.1,
        level_noise_fraction: float = 0.01,
        trend_noise_fraction: float = 0.1,
        initial_variance: float = 1e6,
    ) -> None:
        if not 0.0 < obs_noise_fraction < 1.0:
            raise ValueError("obs_noise_fraction must be in (0, 1)")
        self.obs_noise_fraction = obs_noise_fraction
        self.level_noise_fraction = level_noise_fraction
        self.trend_noise_fraction = trend_noise_fraction
        self.initial_variance = initial_variance
        self.n_channels: Optional[int] = None
        self.state: Optional[np.ndarray] = None  # (d, 2): [level, trend]
        self.cov: Optional[np.ndarray] = None  # (d, 2, 2)
        self.Q: Optional[np.ndarray] = None  # (d, 2, 2), diagonal process noise
        self.R: Optional[np.ndarray] = None  # (d,), observation noise variance

    def fit(self, X: np.ndarray) -> "KalmanLocalTrend":
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        self.n_channels = d
        var = X.var(axis=0, ddof=1)
        var[var == 0] = 1.0
        diff_var = np.diff(X, axis=0).var(axis=0, ddof=1) if X.shape[0] > 1 else var
        diff_var[diff_var == 0] = var[diff_var == 0]

        self.R = self.obs_noise_fraction * var
        self.Q = np.zeros((d, 2, 2))
        self.Q[:, 0, 0] = self.level_noise_fraction * diff_var
        self.Q[:, 1, 1] = self.trend_noise_fraction * self.level_noise_fraction * diff_var

        self.state = np.zeros((d, 2))
        self.state[:, 0] = X[0]
        self.cov = np.zeros((d, 2, 2))
        self.cov[:, 0, 0] = self.initial_variance
        self.cov[:, 1, 1] = self.initial_variance

        self.score(X)  # run the filter forward through the fit batch to warm up the state
        return self

    def partial_fit(self, X: np.ndarray) -> "KalmanLocalTrend":
        if self.state is None:
            return self.fit(X)
        self.score(np.asarray(X, dtype=np.float64))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("KalmanLocalTrend must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        out = np.empty((n, d))
        # Transition matrix F = [[1, 1], [0, 1]] applied per channel: level += trend.
        for t in range(n):
            # Predict.
            level_pred = self.state[:, 0] + self.state[:, 1]
            trend_pred = self.state[:, 1]
            p00, p01, p11 = self.cov[:, 0, 0], self.cov[:, 0, 1], self.cov[:, 1, 1]
            # P_pred = F P F^T + Q, F = [[1,1],[0,1]].
            pp00 = p00 + 2 * p01 + p11 + self.Q[:, 0, 0]
            pp01 = p01 + p11
            pp11 = p11 + self.Q[:, 1, 1]

            innovation = X[t] - level_pred
            s = pp00 + self.R  # innovation variance (observing level only)
            out[t] = np.abs(innovation) / np.sqrt(np.maximum(s, 1e-12))

            # Update (Kalman gain onto [level, trend] from the level-only observation).
            k_level = pp00 / s
            k_trend = pp01 / s
            self.state[:, 0] = level_pred + k_level * innovation
            self.state[:, 1] = trend_pred + k_trend * innovation
            self.cov[:, 0, 0] = (1 - k_level) * pp00
            self.cov[:, 0, 1] = (1 - k_level) * pp01
            self.cov[:, 1, 0] = self.cov[:, 0, 1]
            self.cov[:, 1, 1] = pp11 - k_trend * pp01
        return out.max(axis=1)
