"""Adaptive EVT (Extreme Value Theory) threshold with cross-fold smoothing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import genpareto


def _gpd_quantile(calib_scores: np.ndarray, q: float, alpha: float, min_excesses: int) -> Optional[float]:
    """One fold's eq. 2: fit a GPD to the excesses over the q-th percentile, extrapolate
    to the alpha upper-tail quantile. Returns None if there isn't enough tail data or
    the fit is degenerate (mirrors `POTThreshold`'s own fallback-to-empirical-quantile
    logic, reused here rather than re-derived)."""
    u = float(np.percentile(calib_scores, q * 100))
    excesses = calib_scores[calib_scores > u] - u
    n_k = calib_scores.shape[0]
    n_u = excesses.shape[0]
    if n_u < min_excesses:
        return None
    try:
        shape, _, scale = genpareto.fit(excesses, floc=0)
    except Exception:
        return None
    p_u = n_u / n_k
    if abs(shape) < 1e-6:
        t_evt = u - scale * np.log(alpha / p_u)
    else:
        t_evt = u + (scale / shape) * ((p_u / alpha) ** shape - 1)
    if not np.isfinite(t_evt) or t_evt < u:
        return None
    return float(t_evt)


class EVTCalibrator:
    def __init__(self, q: float = 0.98, alpha: float = 0.01, min_excesses: int = 30, lam: float = 0.05, delta: float = 0.05):
        self.q = q
        self.alpha = alpha
        self.min_excesses = min_excesses
        self.lam = lam
        self.delta = delta
        self.prev_threshold: Optional[float] = None

    def calibrate(self, calib_scores: np.ndarray) -> float:
        """Eq. 2-3: this fold's raw EVT quantile (falling back to the empirical
        1-alpha quantile if the GPD fit is unusable), exponentially smoothed against
        the running threshold, then clipped to at most `delta` relative change."""
        calib_scores = np.asarray(calib_scores, dtype=np.float64)
        t_evt = _gpd_quantile(calib_scores, self.q, self.alpha, self.min_excesses)
        if t_evt is None:
            t_evt = float(np.percentile(calib_scores, (1 - self.alpha) * 100))

        if self.prev_threshold is None:
            threshold = t_evt
        else:
            smoothed = self.lam * t_evt + (1 - self.lam) * self.prev_threshold
            lo, hi = (1 - self.delta) * self.prev_threshold, (1 + self.delta) * self.prev_threshold
            threshold = float(np.clip(smoothed, min(lo, hi), max(lo, hi)))
        self.prev_threshold = threshold
        return threshold
