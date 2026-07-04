from datetime import datetime

import polars as pl

from esa_adb.preprocessing.grid import build_target_grid
from esa_adb.preprocessing.telecommands import encode_telecommand_impulses


def test_impulse_is_single_sample_no_smearing():
    tc_times = pl.DataFrame({"time": [datetime(2020, 1, 1, 8, 10, 13)]})
    grid = build_target_grid(datetime(2020, 1, 1, 8, 10, 10), datetime(2020, 1, 1, 8, 10, 40), 10)

    encoded = encode_telecommand_impulses(tc_times, grid, 10, "tc_1")

    assert encoded["tc_1"].to_list() == [1, 0, 0, 0]


def test_empty_telecommands_all_zero():
    tc_times = pl.DataFrame({"time": []}, schema={"time": pl.Datetime})
    grid = build_target_grid(datetime(2020, 1, 1, 8, 10, 10), datetime(2020, 1, 1, 8, 10, 40), 10)

    encoded = encode_telecommand_impulses(tc_times, grid, 10, "tc_1")

    assert encoded["tc_1"].to_list() == [0, 0, 0, 0]
