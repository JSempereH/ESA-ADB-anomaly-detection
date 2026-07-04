from datetime import datetime

import polars as pl

from esa_adb.preprocessing.grid import build_target_grid
from esa_adb.preprocessing.resample import resample_channel_chunk


def _raw(times, values, labels):
    return pl.DataFrame(
        {"time": times, "value": values, "label_code": labels}
    ).with_columns(pl.col("value").cast(pl.Float32), pl.col("label_code").cast(pl.Int8))


def test_zoh_paper_worked_example():
    # <8:10:12, 8:10:14, 8:10:38> -> v1, v2, v3, all nominal.
    times = [datetime(2020, 1, 1, 8, 10, 12), datetime(2020, 1, 1, 8, 10, 14), datetime(2020, 1, 1, 8, 10, 38)]
    raw = _raw(times, [1.0, 2.0, 3.0], [0, 0, 0])
    grid = build_target_grid(datetime(2020, 1, 1, 8, 10, 10), datetime(2020, 1, 1, 8, 10, 40), 10)

    resampled, carry = resample_channel_chunk(raw, grid, None, "channel_1")

    assert resampled["time"].to_list() == grid["time"].to_list()
    # 8:10:10 -> backward-filled v1 (leading gap); 8:10:20 -> v2; 8:10:30 -> still v2; 8:10:40 -> v3.
    assert resampled["channel_1"].to_list() == [1.0, 2.0, 2.0, 3.0]
    assert carry.value == 3.0


def test_correction_pass_rescues_shadowed_annotated_sample():
    # 8:10:14 is annotated (label=1) but a later nominal sample at 8:10:16 would normally
    # win the 8:10:20 bin via plain ZOH, silently dropping the annotation.
    times = [
        datetime(2020, 1, 1, 8, 10, 12),
        datetime(2020, 1, 1, 8, 10, 14),
        datetime(2020, 1, 1, 8, 10, 16),
        datetime(2020, 1, 1, 8, 10, 38),
    ]
    raw = _raw(times, [1.0, 2.0, 2.5, 3.0], [0, 1, 0, 0])
    grid = build_target_grid(datetime(2020, 1, 1, 8, 10, 10), datetime(2020, 1, 1, 8, 10, 40), 10)

    resampled, _ = resample_channel_chunk(raw, grid, None, "channel_1")

    row = resampled.filter(pl.col("time") == datetime(2020, 1, 1, 8, 10, 20)).row(0, named=True)
    assert row["channel_1"] == 2.0
    assert row["channel_1__label"] == 1


def test_chunk_boundary_carry_over_matches_unchunked_run():
    times = [
        datetime(2020, 1, 1, 8, 10, 12),
        datetime(2020, 1, 1, 8, 10, 14),
        datetime(2020, 1, 1, 8, 10, 38),
    ]
    raw = _raw(times, [1.0, 2.0, 3.0], [0, 0, 0])
    grid = build_target_grid(datetime(2020, 1, 1, 8, 10, 10), datetime(2020, 1, 1, 8, 10, 40), 10)

    unchunked, _ = resample_channel_chunk(raw, grid, None, "channel_1")

    split_time = datetime(2020, 1, 1, 8, 10, 20)
    grid1 = grid.filter(pl.col("time") < split_time)
    grid2 = grid.filter(pl.col("time") >= split_time)
    raw1 = raw.filter(pl.col("time") < split_time)
    raw2 = raw.filter(pl.col("time") >= split_time)

    chunk1, carry = resample_channel_chunk(raw1, grid1, None, "channel_1")
    chunk2, _ = resample_channel_chunk(raw2, grid2, carry, "channel_1")

    chunked = pl.concat([chunk1, chunk2], how="vertical")
    assert chunked["time"].to_list() == unchunked["time"].to_list()
    assert chunked["channel_1"].to_list() == unchunked["channel_1"].to_list()
    assert chunked["channel_1__label"].to_list() == unchunked["channel_1__label"].to_list()


def test_leading_gap_backward_fills_from_first_sample():
    times = [datetime(2020, 1, 1, 8, 10, 25)]
    raw = _raw(times, [9.0], [0])
    grid = build_target_grid(datetime(2020, 1, 1, 8, 10, 10), datetime(2020, 1, 1, 8, 10, 30), 10)

    resampled, _ = resample_channel_chunk(raw, grid, None, "channel_1")
    assert resampled["channel_1"].to_list() == [9.0, 9.0, 9.0]
