"""Lagged-ground-truth helpers for two label-lag protocols.

"Day 1 of a new satellite" is the start of a mission, not its whole lifetime.
Real anomalies that happened a few months ago have realistically had time to
be reviewed and confirmed by now. Both protocols below stay strictly causal: a
fold only ever sees labels for events that ended at least `lag` months before
it. This is a distinct supervision tier from this project's usual L0/L1/L2
(see the README): labels do touch training here, just with a realistic delay,
so these results should never be reported next to L2 numbers without saying so.

Two variants:
- revealed_oracle_positives (PU-lag): assumes an external review process
  confirms every real anomaly after the lag, whether or not the model ever
  flagged it.
- AlarmHistory (active-lag): assumes only the alarms the model itself raised
  get reviewed. After the lag each is confirmed true or false positive, and
  both outcomes feed back. Nobody retroactively reviews telemetry nothing
  ever flagged, so this is the more realistic of the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from esa_adb.benchmark import BenchmarkData, _load_slice
from esa_adb.evaluation import anomalies_only
from esa_adb.models.base import channel_matrix
from esa_adb.models.postprocess import _segments


def revealed_oracle_positives(
    data: BenchmarkData, as_of: datetime, lag_months: int, max_rows: int = 5_000
) -> np.ndarray | None:
    """Real anomaly rows whose event ended at least `lag_months` before `as_of`.
    Strictly causal, never looks at events still inside the lag window. Returns
    None if nothing has been revealed yet (the first `lag_months` of a mission)."""
    cutoff = as_of - relativedelta(months=lag_months)
    anomaly_events = anomalies_only(data.events).filter(pl.col("EndTime") <= cutoff)
    if anomaly_events.height == 0:
        return None
    frames = []
    for row in anomaly_events.iter_rows(named=True):
        seg = _load_slice(data, row["StartTime"], row["EndTime"])
        if seg.height > 0:
            frames.append(channel_matrix(seg, data.input_channels))
    if not frames:
        return None
    out = np.concatenate(frames, axis=0)
    return _subsample(out, max_rows)


def _subsample(rows: np.ndarray, max_rows: int, seed: int = 42) -> np.ndarray:
    if rows.shape[0] <= max_rows:
        return rows
    idx = np.random.default_rng(seed).choice(rows.shape[0], size=max_rows, replace=False)
    return rows[idx]


@dataclass
class _PendingFold:
    fold_index: int
    eval_start: datetime
    eval_end: datetime
    eval_times: np.ndarray
    eval_values: np.ndarray
    pred: np.ndarray


@dataclass
class AlarmHistory:
    """Buffers each fold's own predictions and, once `lag_months` have passed,
    confirms them against the real labels, splitting that fold's alarmed rows
    into confirmed true positives and confirmed false positives. Only alarms
    the model actually raised are ever confirmed, matching real operations."""

    lag_months: int
    pending: list[_PendingFold] = field(default_factory=list)

    def record(self, fold_index: int, eval_start: datetime, eval_end: datetime, eval_times: np.ndarray, eval_values: np.ndarray, pred: np.ndarray) -> None:
        self.pending.append(_PendingFold(fold_index, eval_start, eval_end, eval_times, eval_values, pred))

    def confirm_ready(
        self, data: BenchmarkData, as_of: datetime, max_segment_rows: int = 500, max_rows: int = 5_000
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Confirms every buffered fold whose alarms are now old enough to have
        been reviewed, removes them from the buffer, and returns the pooled
        true-positive and false-positive rows across all of them.

        Each alarm segment is capped at `max_segment_rows` before pooling, or a
        single badly-calibrated fold that alarms for most of a month could dump
        that whole month into the exemplar set as if it were many independent
        examples."""
        ready, still_pending = [], []
        for item in self.pending:
            if item.eval_end + relativedelta(months=self.lag_months) <= as_of:
                ready.append(item)
            else:
                still_pending.append(item)
        self.pending = still_pending
        if not ready:
            return None, None

        anomaly_events = anomalies_only(data.events)
        tp_frames, fp_frames = [], []
        for item in ready:
            starts, ends = _segments(item.pred)
            for seg_start, seg_end in zip(starts, ends):
                seg_times = item.eval_times[seg_start:seg_end]
                # A true positive must overlap a real anomaly specifically, not
                # just any annotated event. Otherwise a rare-but-normal event
                # would count as confirmed and dilute the positive exemplar set.
                overlaps_real_anomaly = _overlaps_any_event(anomaly_events, seg_times[0], seg_times[-1])
                rows = _subsample(item.eval_values[seg_start:seg_end], max_segment_rows)
                (tp_frames if overlaps_real_anomaly else fp_frames).append(rows)

        tp = _subsample(np.concatenate(tp_frames, axis=0), max_rows) if tp_frames else None
        fp = _subsample(np.concatenate(fp_frames, axis=0), max_rows) if fp_frames else None
        return tp, fp


def _overlaps_any_event(events: pl.DataFrame, seg_start: np.datetime64, seg_end: np.datetime64) -> bool:
    hits = events.filter((pl.col("StartTime") <= seg_end) & (pl.col("EndTime") >= seg_start))
    return hits.height > 0
