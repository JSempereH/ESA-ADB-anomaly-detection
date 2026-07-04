from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from esa_adb.preprocessing.windowing import WindowSpec, window_row_slices

FEATURE_NAMES = ["mean", "std", "skew", "kurtosis", "slope", "spectral_energy", "spectral_centroid"]


def _window_features_1d(window: np.ndarray) -> np.ndarray:
    """window: (length, n_channels) -> (len(FEATURE_NAMES) * n_channels,) feature vector."""
    length, n_channels = window.shape

    mean = window.mean(axis=0)
    std = window.std(axis=0)
    skewness = skew(window, axis=0)
    kurt = kurtosis(window, axis=0)

    t = np.arange(length, dtype=np.float64)
    t_centered = t - t.mean()
    denom = (t_centered**2).sum()
    slope = (t_centered[:, None] * (window - mean)).sum(axis=0) / denom if denom > 0 else np.zeros(n_channels)

    fft_mag = np.abs(np.fft.rfft(window - mean, axis=0))  # remove DC so energy reflects variation, not offset
    spectral_energy = (fft_mag**2).sum(axis=0)
    freqs = np.fft.rfftfreq(length)
    total_mag = fft_mag.sum(axis=0)
    spectral_centroid = np.divide(
        (fft_mag * freqs[:, None]).sum(axis=0), total_mag, out=np.zeros(n_channels), where=total_mag > 0
    )

    return np.concatenate([mean, std, skewness, kurt, slope, spectral_energy, spectral_centroid])


def extract_window_features(values: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Statistical + spectral features per window per channel: mean, std, skewness,
    kurtosis, linear trend slope, spectral energy, spectral centroid -- in the spirit of
    the feature set used by *A Hierarchical Ensemble Pipeline for Anomaly Detection in
    ESA Satellite Telemetry* (Allegrini & Pompei, arXiv:2605.06681), which placed 3rd on
    the ESA-ADB Kaggle challenge private leaderboard. That paper's pipeline is supervised
    (XGBoost trained on anomaly labels) and heavy (10h CPU training) -- both conflict with
    this project's unsupervised, "day 1 of a new satellite" framing -- but its feature set
    itself is lightweight and label-free, so it's reused here as a drop-in alternative to
    raw flattened window values for `WindowedIsolationForest` (`feature_extractor=`).

    A fixed `length x n_channels` feature vector per window, richer than raw values: two
    windows with the same mean/variance but different shapes (a slow drift vs. a sudden
    spike, a smooth signal vs. an oscillating one) look identical to a purely
    value-flattening feature vector but are distinguished by skewness/kurtosis/spectral
    content here.
    """
    n_channels = values.shape[1] if values.ndim > 1 else 1
    rows = [_window_features_1d(values[start:end]) for _, start, end in window_row_slices(values.shape[0], spec)]
    if not rows:
        return np.empty((0, len(FEATURE_NAMES) * n_channels))
    return np.stack(rows)
