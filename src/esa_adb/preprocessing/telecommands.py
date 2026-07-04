from __future__ import annotations

import polars as pl

from esa_adb.preprocessing.grid import floor_to_resolution


def encode_telecommand_impulses(tc_times: pl.DataFrame, grid: pl.DataFrame, freq_seconds: int, name: str) -> pl.DataFrame:
    """Encode telecommand execution timestamps as single-sample binary impulses.

    Each TC execution is snapped to the grid bin it falls in (floor to the target
    resolution) and marked 1 there; every other grid point is 0. Unlike value channels,
    telecommand columns are NOT forward-filled/smeared across subsequent grid points --
    a TC firing once should show up as a single impulse, matching the paper's encoding.
    """
    if tc_times.height == 0:
        return grid.select(pl.col("time"), pl.lit(0).cast(pl.Int8).alias(name))

    binned = tc_times.select(
        pl.col("time").map_elements(lambda t: floor_to_resolution(t, freq_seconds), return_dtype=pl.Datetime).alias("time")
    ).unique()
    binned = binned.with_columns(pl.lit(1).cast(pl.Int8).alias(name))

    result = grid.join(binned, on="time", how="left").with_columns(
        pl.col(name).fill_null(0).cast(pl.Int8)
    )
    return result.select("time", name)
