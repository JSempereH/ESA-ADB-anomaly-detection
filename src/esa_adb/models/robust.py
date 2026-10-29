"""Shared robustness helper used by several detectors."""
from __future__ import annotations

import numpy as np


def mad_filter(X: np.ndarray, k: float = 3.5, max_reject_fraction: float = 0.1) -> np.ndarray:
    """Return a row mask (True = keep) that drops the most extreme rows of a batch.

    Each row's "weirdness" is the mean absolute per-channel z-score, computed with
    the median and MAD instead of the mean and std so a handful of outliers can't
    pull the threshold around. Rows are rejected if their score is too high, but
    never more than `max_reject_fraction` of the batch, so a genuinely unusual
    month doesn't get most of its data thrown away.
    """
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad[mad == 0] = 1.0
    z = np.abs(X - median) / (1.4826 * mad)
    deviation = z.mean(axis=1)
    threshold = np.median(deviation) + k * (np.median(np.abs(deviation - np.median(deviation))) * 1.4826 + 1e-9)
    reject = deviation > threshold
    max_reject = int(max_reject_fraction * X.shape[0])
    if reject.sum() > max_reject:
        order = np.argsort(-deviation, kind="stable")
        reject = np.zeros(X.shape[0], dtype=bool)
        reject[order[:max_reject]] = True
    return ~reject
