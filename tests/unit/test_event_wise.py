from datetime import datetime, timedelta

import numpy as np
import polars as pl

from esa_adb.evaluation import evaluate_event_wise, evaluate_event_wise_corrected


def _times(n, start=datetime(2020, 1, 1)):
    return np.array([start + timedelta(seconds=10 * i) for i in range(n)], dtype="datetime64[us]")


def test_long_true_positive_is_one_detection_not_penalized_per_row():
    # A single long detection spanning an entire real event should NOT be penalized the
    # way a row-level metric would for any individual row within it -- exactly the
    # behavior this metric is meant to differ on.
    row_times = _times(20)
    row_pred = np.zeros(20, dtype=bool)
    row_pred[5:15] = True  # one contiguous 10-row detection

    events = pl.DataFrame(
        {
            "StartTime": [row_times[8].astype("datetime64[us]").item()],
            "EndTime": [row_times[12].astype("datetime64[us]").item()],
        }
    )

    result = evaluate_event_wise(row_pred, row_times, events, beta=0.5)

    assert result["n_detections"] == 1
    assert result["n_events"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f_beta"] == 1.0


def test_fragmented_detections_count_as_multiple_alerts():
    row_times = _times(20)
    row_pred = np.zeros(20, dtype=bool)
    row_pred[2:4] = True
    row_pred[10:12] = True  # two separate contiguous runs -> 2 detections

    events = pl.DataFrame({"StartTime": [row_times[10].item()], "EndTime": [row_times[11].item()]})

    result = evaluate_event_wise(row_pred, row_times, events, beta=0.5)

    assert result["n_detections"] == 2
    assert result["n_detections_hit"] == 1  # only the second run overlaps the real event
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_missed_event_gives_zero_recall():
    row_times = _times(10)
    row_pred = np.zeros(10, dtype=bool)  # nothing flagged at all
    events = pl.DataFrame({"StartTime": [row_times[3].item()], "EndTime": [row_times[5].item()]})

    result = evaluate_event_wise(row_pred, row_times, events, beta=0.5)

    assert result["n_detections"] == 0
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0


def test_no_events_and_no_detections_does_not_crash():
    row_times = _times(5)
    row_pred = np.zeros(5, dtype=bool)
    events = pl.DataFrame({"StartTime": [], "EndTime": []}, schema={"StartTime": pl.Datetime, "EndTime": pl.Datetime})

    result = evaluate_event_wise(row_pred, row_times, events)
    assert result["f_beta"] == 0.0


def test_more_forgiving_than_window_level_for_a_slightly_offset_long_detection():
    # A detection that's mostly-but-not-perfectly aligned with a long real event still
    # counts as a full match here (event-wise, overlap-based), unlike a row-level metric
    # which would count every non-overlapping row as a separate FP/FN.
    row_times = _times(30)
    row_pred = np.zeros(30, dtype=bool)
    row_pred[5:25] = True  # detection is offset/wider than the event but still overlaps it throughout

    events = pl.DataFrame({"StartTime": [row_times[10].item()], "EndTime": [row_times[20].item()]})

    result = evaluate_event_wise(row_pred, row_times, events, beta=0.5)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_flagging_everything_is_a_flaw_evaluate_event_wise_has_and_this_fixes():
    # The exact flaw the paper's TNR_t correction exists for: flag literally every row
    # -> one giant detection trivially overlaps every real event -> plain
    # evaluate_event_wise gives a PERFECT precision (1.0), even though almost the entire
    # otherwise-quiet timeline was a false alarm. evaluate_event_wise_corrected must not.
    row_times = _times(1000)
    row_pred = np.ones(1000, dtype=bool)  # flag everything
    events = pl.DataFrame({"StartTime": [row_times[500].item()], "EndTime": [row_times[510].item()]})

    naive = evaluate_event_wise(row_pred, row_times, events, beta=0.5)
    assert naive["precision"] == 1.0  # the documented flaw

    corrected = evaluate_event_wise_corrected(row_pred, row_times, events, beta=0.5)
    assert corrected["precision"] < 0.05  # TNR_t crushes it -- almost the whole quiet timeline was flagged
    assert corrected["recall"] == naive["recall"]  # the correction never touches recall


def test_corrected_precision_equals_raw_precision_when_nothing_outside_events_is_flagged():
    # A detector that ONLY ever flags rows inside real events (a perfectly quiet
    # "should stay quiet" timeline, TNR_t=1) should get the exact same precision as the
    # uncorrected metric -- the correction is a no-op in this best case.
    row_times = _times(30)
    row_pred = np.zeros(30, dtype=bool)
    row_pred[10:15] = True  # entirely inside the event below

    events = pl.DataFrame({"StartTime": [row_times[8].item()], "EndTime": [row_times[20].item()]})

    naive = evaluate_event_wise(row_pred, row_times, events, beta=0.5)
    corrected = evaluate_event_wise_corrected(row_pred, row_times, events, beta=0.5)

    assert corrected["tnr_t"] == 1.0
    assert corrected["precision"] == naive["precision"]
    assert corrected["recall"] == naive["recall"]


def test_corrected_precision_is_raw_precision_times_tnr_t():
    row_times = _times(100)
    row_pred = np.zeros(100, dtype=bool)
    row_pred[10:15] = True  # hits the event
    row_pred[50:60] = True  # false alarm, outside any event, over otherwise-quiet time

    events = pl.DataFrame({"StartTime": [row_times[10].item()], "EndTime": [row_times[14].item()]})

    result = evaluate_event_wise_corrected(row_pred, row_times, events, beta=0.5)

    # n_t = 100 rows - 5 event rows = 95 nominal rows; tn_t = 95 - 10 falsely flagged = 85
    assert result["n_t"] == 95
    assert result["tn_t"] == 85
    np.testing.assert_allclose(result["tnr_t"], 85 / 95)
    np.testing.assert_allclose(result["precision"], result["raw_precision"] * (85 / 95))


def test_corrected_matches_naive_on_recall_and_event_detection_counts():
    row_times = _times(50)
    row_pred = np.zeros(50, dtype=bool)
    row_pred[5:8] = True
    row_pred[30:33] = True

    events = pl.DataFrame(
        {"StartTime": [row_times[6].item(), row_times[40].item()], "EndTime": [row_times[9].item(), row_times[42].item()]}
    )

    naive = evaluate_event_wise(row_pred, row_times, events, beta=0.5)
    corrected = evaluate_event_wise_corrected(row_pred, row_times, events, beta=0.5)

    assert corrected["n_events"] == naive["n_events"] == 2
    assert corrected["n_events_detected"] == naive["n_events_detected"] == 1
    assert corrected["recall"] == naive["recall"] == 0.5
