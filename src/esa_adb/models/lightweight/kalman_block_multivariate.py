"""Multivariate extension of KalmanLocalTrend.

A plain per-channel Kalman filter can miss an anomaly that breaks the usual
correlation between two channels, if each channel still looks normal on its own.
This version keeps the same per-channel level/trend recursion, but scores
channels in groups (subsystem blocks) using a Mahalanobis distance against each
block's innovation covariance, instead of scoring each channel against only its
own variance. A correlation break then shows up as a large joint distance even
when no single channel looks unusual.

Channels outside any block fall back to the plain per-channel score.

The block covariance is estimated once from the fit batch and kept fixed. We do
not adapt it online, because an unfiltered fit batch can contain real anomalies,
and an online update with no gate could teach the filter to treat those as
normal.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.robust import mad_filter


class KalmanBlockMultivariate(AnomalyDetector):
    def __init__(
        self,
        channel_groups: dict[str, list[int]],
        obs_noise_fraction: float = 0.1,
        level_noise_fraction: float = 0.01,
        trend_noise_fraction: float = 0.1,
        initial_variance: float = 1e6,
        min_block_size: int = 2,
        block_reg: float = 0.05,
    ) -> None:
        if not 0.0 < obs_noise_fraction < 1.0:
            raise ValueError("obs_noise_fraction must be in (0, 1)")
        self.channel_groups = channel_groups
        self.obs_noise_fraction = obs_noise_fraction
        self.level_noise_fraction = level_noise_fraction
        self.trend_noise_fraction = trend_noise_fraction
        self.initial_variance = initial_variance
        self.min_block_size = min_block_size
        self.block_reg = block_reg

        self.n_channels: Optional[int] = None
        self.state: Optional[np.ndarray] = None  # (d, 2): [level, trend]
        self.cov: Optional[np.ndarray] = None  # (d, 2, 2)
        self.Q: Optional[np.ndarray] = None  # (d, 2, 2), diagonal process noise
        self.R: Optional[np.ndarray] = None  # (d,), observation noise variance
        self._blocks: list[np.ndarray] = []  # each a channel-index array, size >= min_block_size
        self._unblocked: Optional[np.ndarray] = None  # channel indices not in any block
        self._block_cov_inv: list[np.ndarray] = []  # one (m, m) inverse per block

    def _init_state(self, X: np.ndarray) -> None:
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

        blocked = set()
        self._blocks = []
        for idx in self.channel_groups.values():
            idx = np.asarray(sorted(set(idx) & set(range(d))), dtype=int)
            if len(idx) >= self.min_block_size:
                self._blocks.append(idx)
                blocked.update(idx.tolist())
        self._unblocked = np.asarray(sorted(set(range(d)) - blocked), dtype=int)

    def _recurse(self, X: np.ndarray, fit_block_cov: bool) -> np.ndarray:
        """Runs the per-channel local-trend recursion over X, returns the raw (not yet
        block-Mahalanobis-scored) innovation matrix (n, d). If `fit_block_cov`, also
        estimates and stores each block's innovation covariance inverse from these
        innovations (called once, on the fit batch, before any score is returned)."""
        n, d = X.shape
        innovations = np.empty((n, d))
        for t in range(n):
            level_pred = self.state[:, 0] + self.state[:, 1]
            trend_pred = self.state[:, 1]
            p00, p01, p11 = self.cov[:, 0, 0], self.cov[:, 0, 1], self.cov[:, 1, 1]
            pp00 = p00 + 2 * p01 + p11 + self.Q[:, 0, 0]
            pp01 = p01 + p11
            pp11 = p11 + self.Q[:, 1, 1]

            innovation = X[t] - level_pred
            innovations[t] = innovation
            s = pp00 + self.R
            k_level = pp00 / s
            k_trend = pp01 / s
            self.state[:, 0] = level_pred + k_level * innovation
            self.state[:, 1] = trend_pred + k_trend * innovation
            self.cov[:, 0, 0] = (1 - k_level) * pp00
            self.cov[:, 0, 1] = (1 - k_level) * pp01
            self.cov[:, 1, 0] = self.cov[:, 0, 1]
            self.cov[:, 1, 1] = pp11 - k_trend * pp01

        if fit_block_cov:
            self._block_cov_inv = []
            for idx in self._blocks:
                block_innov = innovations[:, idx]
                m = len(idx)
                # The fit batch is not label-filtered, so a few real anomalies in it
                # can inflate the raw innovation variance by orders of magnitude.
                # Drop the most extreme rows before estimating correlation, the same
                # MAD-based filter used elsewhere in this project.
                keep = mad_filter(block_innov, k=3.5, max_reject_fraction=0.1)
                clean = block_innov[keep] if keep.sum() >= max(2 * m, 30) else block_innov

                # The diagonal (per-channel variance) comes from self.R, the same
                # estimate KalmanLocalTrend already uses, not from the innovation
                # sample variance. A near-constant channel can have innovation
                # variance as low as 1e-11, and Mahalanobis distance scales with
                # 1/variance, so an underestimate there blows up the score. Only the
                # correlation structure, which is bounded between -1 and 1 and can't
                # blow up, is taken from the innovations, shrunk slightly toward
                # independence for conditioning.
                corr = np.corrcoef(clean, rowvar=False) if m > 1 else np.ones((1, 1))
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                corr = (1 - self.block_reg) * corr + self.block_reg * np.eye(m)
                std = np.sqrt(np.maximum(self.R[idx], 1e-30))
                sigma = (std[:, None] * corr) * std[None, :]
                self._block_cov_inv.append(np.linalg.inv(sigma))
        return innovations

    def _score_from_innovations(self, innovations: np.ndarray) -> np.ndarray:
        """A block's squared Mahalanobis distance grows with block size (it follows a
        chi-square distribution with m degrees of freedom), so a raw comparison would
        just favor whichever block happens to be largest. Normalize each block's
        distance to roughly the same scale before taking the row-wise max."""
        n = innovations.shape[0]
        out = np.full(n, -np.inf)
        for idx, sigma_inv in zip(self._blocks, self._block_cov_inv):
            v = innovations[:, idx]  # (n, m)
            m = len(idx)
            d2 = np.einsum("ti,ij,tj->t", v, sigma_inv, v)
            normalized = (d2 - m) / np.sqrt(2.0 * m)
            out = np.maximum(out, normalized)
        if len(self._unblocked):
            per_channel = np.abs(innovations[:, self._unblocked]) / np.sqrt(np.maximum(self.R[self._unblocked], 1e-12))
            out = np.maximum(out, per_channel.max(axis=1))
        return out

    def fit(self, X: np.ndarray) -> "KalmanBlockMultivariate":
        X = np.asarray(X, dtype=np.float64)
        self._init_state(X)
        self._recurse(X, fit_block_cov=True)
        return self

    def partial_fit(self, X: np.ndarray) -> "KalmanBlockMultivariate":
        if self.state is None:
            return self.fit(X)
        self.score(np.asarray(X, dtype=np.float64))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("KalmanBlockMultivariate must be fit before scoring")
        innovations = self._recurse(np.asarray(X, dtype=np.float64), fit_block_cov=False)
        return self._score_from_innovations(innovations)
