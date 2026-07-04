from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve


def find_best_threshold(scores: np.ndarray, y_true: np.ndarray, beta: float = 0.5) -> dict:
    """Scan candidate thresholds and return the one maximizing F-beta.

    Every detector in this project ships with a threshold picked by an ad hoc rule (an
    n-sigma multiple, a fixed percentile of validation residuals, ...) -- reasonable
    defaults, but not tuned to actually maximize detection quality. This scans
    thresholds using `sklearn`'s `precision_recall_curve` (efficient: O(n log n) via the
    sorted score order, not a naive grid search) and picks the one maximizing F-beta.

    Must be called on **labeled validation data, never the test set** -- picking a
    threshold by scanning the test set's own labels would leak test information into a
    number that's supposed to measure generalization. Using validation labels to select a
    threshold (as opposed to training the detector's parameters on them) is standard,
    legitimate practice for the semi-supervised detectors in this project: `fit()` only
    ever sees nominal data, so the detector itself is never trained on anomaly labels,
    only the threshold choice is informed by them.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # drop the threshold=+inf, recall=0 sentinel pair
    if thresholds.size == 0:
        return {"threshold": float("inf"), "precision": 0.0, "recall": 0.0, "f_beta": 0.0, "beta": beta}

    beta2 = beta**2
    denom = beta2 * precision + recall
    f_beta = np.divide((1 + beta2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    best_idx = int(np.argmax(f_beta))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f_beta": float(f_beta[best_idx]),
        "beta": beta,
    }


def evaluate_window_predictions(predicted: np.ndarray, labelled_bounds: pl.DataFrame, beta: float = 0.5) -> dict:
    """Precision/recall/F-beta of window-level anomaly predictions against ground truth.

    `predicted`: bool array, one entry per window, ordered by `window_index` starting at 0
    (i.e. exactly the order `preprocessing.windowing.window_row_slices`/`generate_window_bounds`
    produce). `labelled_bounds`: the output of `preprocessing.windowing.label_windows` --
    a window counts as a ground-truth positive if it isn't nominal (`is_nominal == False`).

    This is a simple window-level detection metric, not the paper's full corrected
    event-wise / affiliation-based / ADTQC metric suite (which needs point-level event
    reconstruction across the whole timeline and is a substantially larger undertaking).
    Good enough to compare lightweight vs. heavy models against each other here, but these
    numbers are not comparable to the numbers published in the ESA-ADB paper.
    """
    y_true = ~labelled_bounds.sort("window_index")["is_nominal"].to_numpy()
    y_pred = np.asarray(predicted, dtype=bool)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"predicted has {y_pred.shape[0]} windows but labelled_bounds has {y_true.shape[0]}")

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta**2
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "beta": beta,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_windows": int(y_true.shape[0]),
        "n_positive": int(y_true.sum()),
    }


def evaluate_events(row_pred: np.ndarray, row_times: np.ndarray, events: pl.DataFrame) -> pl.DataFrame:
    """Per-event detection report: for each real annotated event overlapping the
    evaluated time range, did the model flag *any* row within [StartTime, EndTime]?

    `events` is the anomalies table built by `io.anomalies.build_anomalies_table`
    (columns include `ID`, `Channel`, `Category`, `StartTime`, `EndTime`). This answers
    "did we catch the anomalies actually marked in the dataset", which
    `evaluate_window_predictions`'s aggregate precision/recall cannot: a long event
    spanning many windows only needs to overlap one flagged row to count as detected
    here, whereas the window-level metric penalizes every individual window within it
    that wasn't flagged.

    This checks detection across *all* evaluated channels together (a row is "flagged"
    regardless of which channel's forecast/score tripped it), not per-channel
    attribution - the same simplification `any_target_anomaly` already makes elsewhere
    in this project. It is also still not the paper's affiliation-based/ADTQC scoring.
    """
    row_times = np.asarray(row_times)
    row_pred = np.asarray(row_pred, dtype=bool)
    if row_times.shape[0] != row_pred.shape[0]:
        raise ValueError(f"row_pred has {row_pred.shape[0]} rows but row_times has {row_times.shape[0]}")

    eval_start, eval_end = row_times.min(), row_times.max()
    in_range = events.filter((pl.col("EndTime") >= eval_start) & (pl.col("StartTime") <= eval_end))
    if in_range.height == 0:
        return in_range.with_columns(pl.lit(False, dtype=pl.Boolean).alias("detected"))

    flagged_times = np.sort(row_times[row_pred])
    detected = []
    for row in in_range.iter_rows(named=True):
        start, end = np.datetime64(row["StartTime"]), np.datetime64(row["EndTime"])
        idx = np.searchsorted(flagged_times, start, side="left")
        detected.append(bool(idx < len(flagged_times) and flagged_times[idx] <= end))

    return in_range.with_columns(pl.Series("detected", detected, dtype=pl.Boolean))


def _contiguous_segments(row_pred: np.ndarray, row_times: np.ndarray) -> list[tuple]:
    """(start_time, end_time) for every maximal run of consecutive flagged rows --
    one real-world "alert" per contiguous run, matching how a deployed detector would
    actually raise alarms, rather than counting every individual flagged row separately.

    Vectorized via edge detection on the boolean array's diff (rising edge = alert
    starts, falling edge = alert ends) instead of a per-row Python loop, since a
    per-sample loop over a multi-million-row array is prohibitively slow (see
    `models/dynamic_threshold.py`'s docstring).
    """
    if not row_pred.any():
        return []
    padded = np.concatenate([[False], row_pred, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(row_times[starts], row_times[ends]))


def evaluate_event_wise(row_pred: np.ndarray, row_times: np.ndarray, events: pl.DataFrame, beta: float = 0.5) -> dict:
    """Event-level precision/recall/F-beta -- closer in spirit to the paper's "corrected
    event-wise" scoring than `evaluate_window_predictions`'s row/window-level metric,
    though still an approximation: it's missing the paper's affiliation-based partial
    credit and its "0.5 precision for empty detections" rule (see README's "Known
    approximations").

    - **Detections**: contiguous runs of flagged rows collapsed into one alert per run
      (`_contiguous_segments`) -- not one true/false positive per individual flagged
      row/window, which is what makes this different from, and generally more forgiving
      than, `evaluate_window_predictions`.
    - **Event-wise recall**: fraction of real events with >=1 overlapping detection
      (the same notion `evaluate_events`/`summarize_event_detection` already compute,
      just aggregated into a single number here instead of a per-category breakdown).
    - **Event-wise precision**: fraction of *detections* (not individual rows) that
      overlap >=1 real event -- a single long true detection that happens to span a real
      event no longer gets penalized point-by-point the way the row-level metric would.
    """
    row_pred = np.asarray(row_pred, dtype=bool)
    row_times = np.asarray(row_times)
    eval_start, eval_end = row_times.min(), row_times.max()
    in_range_events = events.filter((pl.col("EndTime") >= eval_start) & (pl.col("StartTime") <= eval_end))
    segments = _contiguous_segments(row_pred, row_times)

    n_events = in_range_events.height
    n_detections = len(segments)
    if n_events == 0 and n_detections == 0:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0, "beta": beta, "n_events": 0, "n_detections": 0}

    event_starts = in_range_events["StartTime"].to_numpy() if n_events else np.array([])
    event_ends = in_range_events["EndTime"].to_numpy() if n_events else np.array([])

    event_detected = np.zeros(n_events, dtype=bool)
    segment_hit = np.zeros(n_detections, dtype=bool)
    for i, (s_start, s_end) in enumerate(segments):
        if n_events == 0:
            break
        overlap = (event_starts <= s_end) & (event_ends >= s_start)
        if overlap.any():
            segment_hit[i] = True
            event_detected |= overlap

    recall = float(event_detected.mean()) if n_events else 0.0
    precision = float(segment_hit.mean()) if n_detections else 0.0
    beta2 = beta**2
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "beta": beta,
        "n_events": n_events,
        "n_detections": n_detections,
        "n_events_detected": int(event_detected.sum()),
        "n_detections_hit": int(segment_hit.sum()),
    }


def _nominal_row_mask(row_times: np.ndarray, in_range_events: pl.DataFrame) -> np.ndarray:
    """True for every row outside all annotated events' [StartTime, EndTime] spans --
    the "should stay quiet" timeline `evaluate_event_wise_corrected`'s TNR_t term needs.
    A Python loop over events (never more than a few hundred, even across a full
    mission), each doing one vectorized row-range update -- not a
    rows x events broadcast, which would be needlessly large for a multi-million-row
    evaluation period.
    """
    nominal = np.ones(row_times.shape[0], dtype=bool)
    for row in in_range_events.iter_rows(named=True):
        start, end = np.datetime64(row["StartTime"]), np.datetime64(row["EndTime"])
        nominal &= ~((row_times >= start) & (row_times <= end))
    return nominal


def evaluate_event_wise_corrected(row_pred: np.ndarray, row_times: np.ndarray, events: pl.DataFrame, beta: float = 0.5) -> dict:
    """The ESA-ADB paper's own "corrected event-wise F-score" (Kotowski et al. 2024,
    arXiv:2406.17826, "Metrics and hierarchical evaluation" / "Corrected event-wise
    F-score", equations 1-2) -- Sehili & Zhang's fix for a real flaw in classic
    event-wise precision: an algorithm that flags literally every sample gets a
    PERFECT classic event-wise precision (one giant detection spanning the whole
    timeline trivially overlaps every real event), since that metric only ever checks
    whether flagged segments touch a real event, never how much of the otherwise-quiet
    timeline got dragged into a false alarm along the way. `evaluate_event_wise` above
    has exactly this flaw (see `test_flagging_everything_is_a_flaw_evaluate_event_wise_has_and_this_fixes`).

    The fix multiplies precision by the time-domain true negative rate:

        Pr_corr = (TP_e / (TP_e + FP_e)) * TNR_t,   TNR_t = TN_t / N_t             (eq. 1)
        F_beta_corr = (1+beta^2) * Pr_corr*Rec_e / (beta^2*Pr_corr + Rec_e)         (eq. 2)

    `TP_e`/`FP_e` (detections that did/didn't hit a real event) and `Rec_e` (fraction
    of real events with >=1 overlapping detection) are exactly `evaluate_event_wise`'s
    own `precision`/`recall` -- reused here, not recomputed -- so this correction only
    ever changes precision, never recall. `N_t`/`TN_t` are counted in TIME, not
    detections: `N_t` is every row outside any annotated event's span (the "should stay
    quiet" timeline, via `_nominal_row_mask`), `TN_t` is however much of that span was
    correctly left unflagged. The paper counts nanoseconds, not rows -- equivalent here
    since this project's preprocessing already resamples every mission onto a single
    uniform time grid (`preprocessing/resample.py`), so row count is directly
    proportional to elapsed time and needs no separate conversion.

    This is the metric behind the paper's own published headline numbers (their Table 4,
    priority aspects 1a/1b) -- comparable to them ONLY when evaluated on the same channel
    subset. Their frequently-cited 0.949 for Windowed iForest on Mission2 is on an easier
    11-channel "lightweight" subset (channels 18-28) -- for the FULL 47-channel set this
    project otherwise reports numbers for, their own Windowed iForest couldn't even run
    (out-of-memory in their pipeline) and their best full-set corrected event-wise F0.5
    is 0.071 (Telemanom-ESA-Pruned) -- confirmed by reading the paper directly, not
    assumed. Beta=0.5 is the paper's own baseline, matching this project's convention
    throughout.
    """
    row_pred = np.asarray(row_pred, dtype=bool)
    row_times = np.asarray(row_times)
    eval_start, eval_end = row_times.min(), row_times.max()
    in_range_events = events.filter((pl.col("EndTime") >= eval_start) & (pl.col("StartTime") <= eval_end))

    base = evaluate_event_wise(row_pred, row_times, events, beta=beta)

    nominal = _nominal_row_mask(row_times, in_range_events)
    n_t = int(nominal.sum())
    tn_t = int((nominal & ~row_pred).sum())
    tnr_t = tn_t / n_t if n_t > 0 else 0.0

    precision_corr = base["precision"] * tnr_t
    recall = base["recall"]
    beta2 = beta**2
    f_beta_corr = (
        (1 + beta2) * precision_corr * recall / (beta2 * precision_corr + recall) if (precision_corr + recall) > 0 else 0.0
    )

    return {
        "precision": precision_corr,
        "recall": recall,
        "f_beta": f_beta_corr,
        "beta": beta,
        "tnr_t": tnr_t,
        "n_t": n_t,
        "tn_t": tn_t,
        "raw_precision": base["precision"],
        "n_events": base["n_events"],
        "n_detections": base["n_detections"],
        "n_events_detected": base["n_events_detected"],
        "n_detections_hit": base["n_detections_hit"],
    }


def summarize_event_detection(events_with_detected: pl.DataFrame) -> pl.DataFrame:
    """Detection rate per event Category, plus an overall "ALL" row."""
    if events_with_detected.height == 0:
        return pl.DataFrame(
            {"Category": ["ALL"], "n_events": [0], "n_detected": [0], "detection_rate": [None]},
            schema={"Category": pl.Utf8, "n_events": pl.UInt32, "n_detected": pl.UInt32, "detection_rate": pl.Float64},
        )

    per_category = events_with_detected.group_by("Category").agg(
        pl.len().alias("n_events"), pl.col("detected").sum().alias("n_detected")
    )
    overall = events_with_detected.select(
        pl.lit("ALL").alias("Category"), pl.len().alias("n_events"), pl.col("detected").sum().alias("n_detected")
    )
    return pl.concat([per_category, overall], how="vertical").with_columns(
        (pl.col("n_detected") / pl.col("n_events")).alias("detection_rate")
    )
