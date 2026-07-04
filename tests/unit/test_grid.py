from datetime import datetime

from esa_adb.preprocessing.grid import build_target_grid, ceil_to_resolution, floor_to_resolution, grid_slice_for_chunk


def test_floor_ceil_paper_worked_example():
    # Paper: raw timestamps <8:10:12, 8:10:14, 8:10:38> resampled to 1/10Hz (10s)
    # should give grid <8:10:10, 8:10:20, 8:10:30, 8:10:40>.
    assert floor_to_resolution(datetime(2020, 1, 1, 8, 10, 12), 10) == datetime(2020, 1, 1, 8, 10, 10)
    assert ceil_to_resolution(datetime(2020, 1, 1, 8, 10, 38), 10) == datetime(2020, 1, 1, 8, 10, 40)


def test_ceil_on_exact_multiple_is_identity():
    assert ceil_to_resolution(datetime(2020, 1, 1, 8, 10, 40), 10) == datetime(2020, 1, 1, 8, 10, 40)


def test_build_target_grid_paper_worked_example():
    start = floor_to_resolution(datetime(2020, 1, 1, 8, 10, 12), 10)
    end = ceil_to_resolution(datetime(2020, 1, 1, 8, 10, 38), 10)
    grid = build_target_grid(start, end, 10)

    expected = [
        datetime(2020, 1, 1, 8, 10, 10),
        datetime(2020, 1, 1, 8, 10, 20),
        datetime(2020, 1, 1, 8, 10, 30),
        datetime(2020, 1, 1, 8, 10, 40),
    ]
    assert grid["time"].to_list() == expected


def test_grid_slice_for_chunk_tiles_without_gaps_or_duplicates():
    global_start = datetime(2020, 1, 1, 0, 0, 0)
    freq = 30
    chunk_a = grid_slice_for_chunk(global_start, datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 0, 10, 0), freq)
    chunk_b = grid_slice_for_chunk(
        global_start, datetime(2020, 1, 1, 0, 10, 0), datetime(2020, 1, 1, 0, 20, 0), freq, include_end=True
    )

    times_a = chunk_a["time"].to_list()
    times_b = chunk_b["time"].to_list()

    assert set(times_a).isdisjoint(set(times_b))
    combined = sorted(times_a + times_b)
    assert combined[0] == datetime(2020, 1, 1, 0, 0, 0)
    assert combined[-1] == datetime(2020, 1, 1, 0, 20, 0)
    # No gaps: consecutive points exactly `freq` seconds apart.
    for a, b in zip(combined, combined[1:]):
        assert (b - a).total_seconds() == freq
