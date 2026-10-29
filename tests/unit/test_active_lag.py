from datetime import datetime

import numpy as np
import polars as pl

from esa_adb.models.active_lag import AlarmHistory


class _FakeData:
    def __init__(self, events: pl.DataFrame):
        self.events = events


def _events(rows):
    return pl.DataFrame(
        {"StartTime": [r[0] for r in rows], "EndTime": [r[1] for r in rows], "Category": [r[2] for r in rows]}
    ).with_columns(pl.col("StartTime").cast(pl.Datetime), pl.col("EndTime").cast(pl.Datetime))


def test_confirm_ready_waits_for_the_lag():
    history = AlarmHistory(lag_months=1)
    times = np.array([datetime(2000, 1, 1, h) for h in range(10)], dtype="datetime64[us]")
    values = np.random.default_rng(0).normal(size=(10, 2))
    pred = np.zeros(10, dtype=bool)
    pred[1:3] = True
    history.record(0, datetime(2000, 1, 1), datetime(2000, 1, 1, 10), times, values, pred)

    data = _FakeData(_events([(datetime(2000, 1, 1, 1), datetime(2000, 1, 1, 2), "anomaly")]))
    tp, fp = history.confirm_ready(data, datetime(2000, 1, 1))
    assert tp is None and fp is None  # not ready yet, lag hasn't passed


def test_confirm_ready_treats_rare_nominal_overlap_as_false_positive():
    """Regression test: an alarm that overlaps a RARE NOMINAL event (not a real
    anomaly) must be confirmed as a false positive, not a true positive -- conflating
    the two would leak rare-nominal-event rows into the positive exemplar set."""
    history = AlarmHistory(lag_months=1)
    times = np.array([datetime(2000, 1, 1, h) for h in range(10)], dtype="datetime64[us]")
    values = np.random.default_rng(0).normal(size=(10, 2))
    pred = np.zeros(10, dtype=bool)
    pred[1:3] = True  # one alarm segment, overlapping a rare-nominal event below
    history.record(0, datetime(2000, 1, 1), datetime(2000, 1, 1, 10), times, values, pred)

    data = _FakeData(_events([(datetime(2000, 1, 1, 1), datetime(2000, 1, 1, 2), "rare nominal event")]))
    tp, fp = history.confirm_ready(data, datetime(2000, 2, 2))
    assert tp is None
    assert fp is not None and fp.shape[0] == 2


def test_confirm_ready_treats_real_anomaly_overlap_as_true_positive():
    history = AlarmHistory(lag_months=1)
    times = np.array([datetime(2000, 1, 1, h) for h in range(10)], dtype="datetime64[us]")
    values = np.random.default_rng(0).normal(size=(10, 2))
    pred = np.zeros(10, dtype=bool)
    pred[1:3] = True
    history.record(0, datetime(2000, 1, 1), datetime(2000, 1, 1, 10), times, values, pred)

    data = _FakeData(_events([(datetime(2000, 1, 1, 1), datetime(2000, 1, 1, 2), "anomaly")]))
    tp, fp = history.confirm_ready(data, datetime(2000, 2, 2))
    assert fp is None
    assert tp is not None and tp.shape[0] == 2


def test_confirm_ready_mixed_segments():
    history = AlarmHistory(lag_months=1)
    times = np.array([datetime(2000, 1, 1, h) for h in range(10)], dtype="datetime64[us]")
    values = np.random.default_rng(0).normal(size=(10, 2))
    pred = np.zeros(10, dtype=bool)
    pred[1:3] = True  # overlaps the real anomaly
    pred[6:8] = True  # overlaps nothing -> false positive
    history.record(0, datetime(2000, 1, 1), datetime(2000, 1, 1, 10), times, values, pred)

    data = _FakeData(_events([(datetime(2000, 1, 1, 1), datetime(2000, 1, 1, 2), "anomaly")]))
    tp, fp = history.confirm_ready(data, datetime(2000, 2, 2))
    assert tp is not None and tp.shape[0] == 2
    assert fp is not None and fp.shape[0] == 2
