from datetime import datetime, timedelta

import numpy as np
import polars as pl

from esa_adb.evaluation import evaluate_events, summarize_event_detection


def _times(n, start=datetime(2020, 1, 1)):
    return np.array([start + timedelta(seconds=10 * i) for i in range(n)], dtype="datetime64[us]")


def test_evaluate_events_detects_event_with_a_single_overlapping_flagged_row():
    row_times = _times(20)
    row_pred = np.zeros(20, dtype=bool)
    row_pred[15] = True  # one flagged row, deep inside the event window below

    events = pl.DataFrame(
        {
            "ID": ["id_1"],
            "Channel": ["channel_1"],
            "Category": ["anomaly"],
            "StartTime": [row_times[10].astype("datetime64[us]").item()],
            "EndTime": [row_times[18].astype("datetime64[us]").item()],
        }
    )

    result = evaluate_events(row_pred, row_times, events)
    assert result["detected"].to_list() == [True]


def test_evaluate_events_misses_event_with_no_overlapping_flagged_row():
    row_times = _times(20)
    row_pred = np.zeros(20, dtype=bool)
    row_pred[0] = True  # flagged, but nowhere near the event

    events = pl.DataFrame(
        {
            "ID": ["id_1"],
            "Channel": ["channel_1"],
            "Category": ["anomaly"],
            "StartTime": [row_times[10].astype("datetime64[us]").item()],
            "EndTime": [row_times[18].astype("datetime64[us]").item()],
        }
    )

    result = evaluate_events(row_pred, row_times, events)
    assert result["detected"].to_list() == [False]


def test_evaluate_events_excludes_events_outside_the_evaluated_range():
    row_times = _times(5)  # only covers indices 0-4
    row_pred = np.zeros(5, dtype=bool)

    events = pl.DataFrame(
        {
            "ID": ["id_1"],
            "Channel": ["channel_1"],
            "Category": ["anomaly"],
            "StartTime": [datetime(2021, 1, 1)],  # way outside row_times' range
            "EndTime": [datetime(2021, 1, 2)],
        }
    )

    result = evaluate_events(row_pred, row_times, events)
    assert result.height == 0


def test_summarize_event_detection_rate_per_category_and_overall():
    events = pl.DataFrame(
        {
            "Category": ["anomaly", "anomaly", "rare event"],
            "detected": [True, False, True],
        }
    )
    summary = summarize_event_detection(events)
    by_category = {row["Category"]: row for row in summary.iter_rows(named=True)}

    assert by_category["anomaly"]["n_events"] == 2
    assert by_category["anomaly"]["n_detected"] == 1
    assert by_category["anomaly"]["detection_rate"] == 0.5
    assert by_category["rare event"]["detection_rate"] == 1.0
    assert by_category["ALL"]["n_events"] == 3
    assert by_category["ALL"]["n_detected"] == 2
