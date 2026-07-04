from datetime import datetime, timedelta

import polars as pl

from esa_adb.preprocessing.windowing import (
    WindowSpec,
    generate_window_bounds,
    label_windows,
    window_bounds_frame,
    window_row_slices,
)

FREQ = 10  # seconds
START = datetime(2020, 1, 1, 0, 0, 0)
END = datetime(2020, 1, 1, 0, 2, 0)  # 2 minutes


def test_generate_window_bounds_overlap_and_tumbling():
    spec = WindowSpec(length=4, stride=2)  # 40s windows, 20s stride -> 50% overlap
    bounds = list(generate_window_bounds(START, END, FREQ, spec))
    assert [b[0] for b in bounds] == list(range(5))
    assert bounds[0][1:] == (datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 0, 0, 40))
    assert bounds[1][1:] == (datetime(2020, 1, 1, 0, 0, 20), datetime(2020, 1, 1, 0, 1, 0))
    assert bounds[-1][1:] == (datetime(2020, 1, 1, 0, 1, 20), datetime(2020, 1, 1, 0, 2, 0))

    tumbling = WindowSpec(length=4, stride=4)  # no overlap
    bounds_tumbling = list(generate_window_bounds(START, END, FREQ, tumbling))
    assert len(bounds_tumbling) == 3  # 120s / 40s
    for i in range(len(bounds_tumbling) - 1):
        assert bounds_tumbling[i][2] == bounds_tumbling[i + 1][1]  # tiles with no gap/overlap


def test_label_windows_flags_all_overlapping_windows_containing_the_event():
    spec = WindowSpec(length=4, stride=2)
    bounds = window_bounds_frame(START, END, FREQ, spec)
    # A label at 00:00:50 falls inside windows [0:20,1:00) and [0:40,1:20) only.
    labels_long = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1, 0, 0, 50)],
            "channel": ["channel_1"],
            "category_code": [1],
        }
    )

    result = label_windows(labels_long, bounds, START, FREQ, spec)
    non_nominal = result.filter(~pl.col("is_nominal"))["window_index"].sort().to_list()

    assert non_nominal == [1, 2]
    assert result.filter(pl.col("window_index") == 1)["channels_affected"].item().to_list() == ["channel_1"]
    assert result.filter(pl.col("window_index") == 0)["is_nominal"].item() is True


def test_label_windows_all_nominal_when_no_labels():
    spec = WindowSpec(length=4, stride=4)
    bounds = window_bounds_frame(START, END, FREQ, spec)
    labels_long = pl.DataFrame({"time": [], "channel": [], "category_code": []})

    result = label_windows(labels_long, bounds, START, FREQ, spec)
    assert result["is_nominal"].all()


def test_label_windows_respects_channel_filter():
    spec = WindowSpec(length=4, stride=4)
    bounds = window_bounds_frame(START, END, FREQ, spec)
    labels_long = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1, 0, 0, 10)],
            "channel": ["non_target_channel"],
            "category_code": [1],
        }
    )

    result_all = label_windows(labels_long, bounds, START, FREQ, spec)
    result_target_only = label_windows(labels_long, bounds, START, FREQ, spec, channels=["channel_1"])

    assert not result_all.filter(pl.col("window_index") == 0)["is_nominal"].item()
    assert result_target_only["is_nominal"].all()  # label's channel isn't in the target filter


def test_window_bounds_frame_and_window_row_slices_agree_on_count():
    # `end` must be an *exclusive* upper bound one grid step past the last row (matching
    # how `config.grid.end` is used elsewhere) for the datetime-based and row-count-based
    # window generators to produce the same number of windows over the same data -- this
    # is exactly the pairing `models/` uses to align window-level labels with features.
    n_rows = 37
    end = START + timedelta(seconds=n_rows * FREQ)
    for spec in [WindowSpec(length=4, stride=2), WindowSpec(length=5, stride=5), WindowSpec(length=10, stride=3)]:
        bounds = window_bounds_frame(START, end, FREQ, spec)
        row_slices = list(window_row_slices(n_rows, spec))
        assert bounds.height == len(row_slices), f"mismatch for {spec}"
        assert bounds["window_index"].to_list() == [i for i, _, _ in row_slices]
