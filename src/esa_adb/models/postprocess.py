from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import lfilter


def ewma(scores: np.ndarray, alpha: Optional[float]) -> np.ndarray:
    """Exponentially weighted moving average of a score series (`alpha=None` or `1.0`
    -> passthrough). Smoothing before thresholding suppresses single-sample score
    spikes, the dominant source of one-row false alarms; the cost is a small detection
    delay on real events (a genuine step in the score needs ~1/alpha samples to fully
    propagate). Telemanom (Hundman et al., KDD 2018) applies the same idea to forecast
    errors before its own thresholding.

    Implemented with `scipy.signal.lfilter` (the EWMA recursion y[t] = alpha*x[t] +
    (1-alpha)*y[t-1] IS a first-order IIR filter) rather than a Python loop, since
    eval periods here run to millions of rows. `zi` seeds the filter state so
    y[0] == x[0] instead of decaying up from zero.
    """
    if alpha is None or alpha >= 1.0:
        return scores
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] == 0:
        return scores
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], scores, zi=[(1.0 - alpha) * scores[0]])
    return y


def _segments(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) index pairs (end exclusive) of every contiguous True run."""
    padded = np.concatenate([[False], pred, [False]])
    diffs = np.diff(padded.astype(np.int8))
    return np.where(diffs == 1)[0], np.where(diffs == -1)[0]


def merge_short_gaps(pred: np.ndarray, max_gap_rows: int) -> np.ndarray:
    """Fill False gaps of <= `max_gap_rows` between consecutive alarm segments.

    A detector whose score hovers around the threshold during one long real event
    otherwise emits a burst of short alarms: one of them overlaps the event (1 TP
    detection) and the rest land in the gaps around it or count as separate detections
    -- event-wise precision pays for every extra fragment. Fusing near-contiguous
    fragments into one alarm is how a human operator would read them anyway.
    """
    pred = np.asarray(pred, dtype=bool)
    starts, ends = _segments(pred)
    if len(starts) < 2:
        return pred.copy()
    out = pred.copy()
    gap_fill = (starts[1:] - ends[:-1]) <= max_gap_rows
    for i in np.where(gap_fill)[0]:
        out[ends[i] : starts[i + 1]] = True
    return out


def enforce_min_duration(pred: np.ndarray, min_rows: int) -> np.ndarray:
    """Drop alarm segments shorter than `min_rows` rows.

    Single-row (or few-row) score spikes are overwhelmingly false alarms on this data,
    while the annotated events span minutes to days -- a minimum-persistence rule
    trades a negligible amount of event-wise recall for a direct cut to FP detections.
    Must run AFTER `merge_short_gaps`, so fragments of one real alarm are fused into a
    long-enough segment before anything is discarded.
    """
    pred = np.asarray(pred, dtype=bool)
    starts, ends = _segments(pred)
    out = pred.copy()
    for s, e in zip(starts, ends):
        if e - s < min_rows:
            out[s:e] = False
    return out


@dataclass(frozen=True)
class PostprocessSpec:
    """Alarm post-processing configuration, in ROWS of the mission's uniform time grid
    (Mission2: 18s/row, so the defaults are: fuse alarms separated by <30 min, then
    drop alarms shorter than 3 min). `smooth_alphas` are the EWMA candidates threshold
    calibration is allowed to choose between per fold (None = no smoothing) -- chosen on
    calibration data, never on the eval period.
    """

    merge_gap_rows: int = 100
    min_duration_rows: int = 10
    smooth_alphas: tuple[Optional[float], ...] = (None, 0.3)

    def apply(self, pred: np.ndarray) -> np.ndarray:
        return enforce_min_duration(merge_short_gaps(pred, self.merge_gap_rows), self.min_duration_rows)
