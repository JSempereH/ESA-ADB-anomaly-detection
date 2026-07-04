from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl

_EPOCH = datetime(1970, 1, 1)


def floor_to_resolution(ts: datetime, freq_seconds: int) -> datetime:
    total = (ts - _EPOCH).total_seconds()
    floored = total - (total % freq_seconds)
    return _EPOCH + timedelta(seconds=floored)


def ceil_to_resolution(ts: datetime, freq_seconds: int) -> datetime:
    floored = floor_to_resolution(ts, freq_seconds)
    return floored if floored == ts else floored + timedelta(seconds=freq_seconds)


def _empty_grid() -> pl.DataFrame:
    return pl.DataFrame({"time": []}, schema={"time": pl.Datetime})


def build_target_grid(start: datetime, end: datetime, freq_seconds: int) -> pl.DataFrame:
    if end < start:
        return _empty_grid()
    times = pl.datetime_range(start, end, interval=f"{freq_seconds}s", eager=True, closed="both")
    return pl.DataFrame({"time": times})


def grid_slice_for_chunk(
    global_start: datetime,
    chunk_start: datetime,
    chunk_end: datetime,
    freq_seconds: int,
    include_end: bool = False,
) -> pl.DataFrame:
    """Grid points offset from ``global_start`` that fall within a chunk window.

    Offsets are always derived from ``global_start`` (never re-floored per chunk) so
    adjacent chunks tile exactly with no duplicate or missing grid points at the seams.
    ``chunk_end`` is treated as exclusive unless ``include_end`` is True (use this only
    for the final chunk, whose end is the frozen ``global_end``).
    """
    k0 = math.ceil((chunk_start - global_start).total_seconds() / freq_seconds)
    raw_k1 = (chunk_end - global_start).total_seconds() / freq_seconds
    if include_end:
        k1 = math.floor(raw_k1 + 1e-9)
    else:
        k1 = math.ceil(raw_k1 - 1e-9) - 1

    if k1 < k0:
        return _empty_grid()

    start_dt = global_start + timedelta(seconds=k0 * freq_seconds)
    end_dt = global_start + timedelta(seconds=k1 * freq_seconds)
    return build_target_grid(start_dt, end_dt, freq_seconds)
