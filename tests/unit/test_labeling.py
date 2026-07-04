from datetime import datetime

import polars as pl

from esa_adb.io.anomalies import attach_raw_labels
from esa_adb.preprocessing.labeling import (
    CategoryCode,
    any_target_anomaly_expr,
    category_to_code,
    extract_labels_long,
    filter_nominal_rows,
    row_is_non_nominal,
)


def test_category_to_code_covers_all_four_categories():
    assert category_to_code("anomaly") == CategoryCode.ANOMALY
    assert category_to_code("rare nominal event") == CategoryCode.RARE_NOMINAL_EVENT
    assert category_to_code("communication gap") == CategoryCode.COMMUNICATION_GAP
    assert category_to_code("invalid segment") == CategoryCode.INVALID_SEGMENT
    assert category_to_code("  Anomaly ") == CategoryCode.ANOMALY  # strip/lowercase


def test_attach_raw_labels_no_cross_channel_leakage():
    # channel_A has an anomaly window; channel_B does not, and must stay NOMINAL
    # at the exact same timestamps -- an anomaly window must never leak across channels.
    anomalies = pl.DataFrame(
        {
            "Channel": ["channel_A"],
            "StartTime": [datetime(2020, 1, 1, 0, 0, 10)],
            "EndTime": [datetime(2020, 1, 1, 0, 0, 20)],
            "category_code": [int(CategoryCode.ANOMALY)],
        }
    )
    times = [datetime(2020, 1, 1, 0, 0, t) for t in (5, 10, 15, 20, 25)]

    raw_a = pl.DataFrame({"time": times, "value": [1.0] * 5}).with_columns(pl.col("value").cast(pl.Float32))
    raw_b = pl.DataFrame({"time": times, "value": [2.0] * 5}).with_columns(pl.col("value").cast(pl.Float32))

    labelled_a = attach_raw_labels(raw_a, anomalies, "channel_A")
    labelled_b = attach_raw_labels(raw_b, anomalies, "channel_B")

    assert labelled_a["label_code"].to_list() == [0, 1, 1, 1, 0]
    assert labelled_b["label_code"].to_list() == [0, 0, 0, 0, 0]


def test_extract_labels_long_only_keeps_non_nominal_rows():
    df = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1, 0, 0, t) for t in (0, 10, 20)],
            "channel_1__label": [0, 2, 0],
        }
    ).rename({"channel_1__label": "label_code"})

    long = extract_labels_long(df, "channel_1")

    assert long["channel"].to_list() == ["channel_1"]
    assert long["category_code"].to_list() == [2]


def test_any_target_anomaly_expr_is_or_across_target_channels():
    df = pl.DataFrame(
        {
            "channel_A__label": [0, 1, 0],
            "channel_B__label": [0, 0, 1],
            "channel_C__label": [2, 0, 0],  # non-target channel, excluded from the OR
        }
    )
    result = df.select(any_target_anomaly_expr(["channel_A__label", "channel_B__label"]))
    assert result["any_target_anomaly"].to_list() == [False, True, True]


def test_filter_nominal_rows_excludes_all_four_categories_not_just_anomaly():
    # any_target_anomaly only catches CategoryCode.ANOMALY -- filter_nominal_rows must
    # also exclude rare nominal events / comm gaps / invalid segments, since those are
    # still "non-nominal" per label_windows' ground truth and shouldn't leak into a
    # semi-supervised model's "nominal-only" training data.
    times = [datetime(2020, 1, 1, 0, 0, t) for t in (0, 10, 20, 30, 40)]
    df = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    labels_long = pl.DataFrame(
        {
            "time": [times[1], times[3]],
            "channel": ["channel_1", "channel_1"],
            "category_code": [int(CategoryCode.RARE_NOMINAL_EVENT), int(CategoryCode.COMMUNICATION_GAP)],
        }
    )

    filtered = filter_nominal_rows(df, labels_long, ["channel_1"])

    assert filtered["time"].to_list() == [times[0], times[2], times[4]]


def test_filter_nominal_rows_ignores_labels_on_other_channels():
    times = [datetime(2020, 1, 1, 0, 0, t) for t in (0, 10)]
    df = pl.DataFrame({"time": times, "value": [1.0, 2.0]})
    labels_long = pl.DataFrame(
        {"time": [times[1]], "channel": ["channel_other"], "category_code": [int(CategoryCode.ANOMALY)]}
    )

    filtered = filter_nominal_rows(df, labels_long, ["channel_1"])

    assert filtered["time"].to_list() == times


def test_row_is_non_nominal_is_the_row_level_complement_of_filter_nominal_rows():
    times = [datetime(2020, 1, 1, 0, 0, t) for t in (0, 10, 20, 30, 40)]
    df = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    labels_long = pl.DataFrame(
        {
            "time": [times[1], times[3]],
            "channel": ["channel_1", "channel_1"],
            "category_code": [int(CategoryCode.RARE_NOMINAL_EVENT), int(CategoryCode.COMMUNICATION_GAP)],
        }
    )

    flags = row_is_non_nominal(df, labels_long, ["channel_1"])

    assert flags.tolist() == [False, True, False, True, False]
    # Rows kept by filter_nominal_rows are exactly the rows NOT flagged here.
    filtered = filter_nominal_rows(df, labels_long, ["channel_1"])
    assert filtered["time"].to_list() == [t for t, f in zip(times, flags) if not f]
