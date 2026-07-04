from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import polars as pl


@dataclass(frozen=True)
class WindowSpec:
    """A sliding window over the resampled grid, expressed in samples (not seconds).

    ``stride < length`` gives overlapping windows (e.g. stride = length // 2 for 50%
    overlap) -- typical for streaming/production-style anomaly detectors that re-score
    on every new arriving sample or small batch rather than waiting for a full new
    non-overlapping window. ``stride == length`` gives non-overlapping (tumbling) windows.
    """

    length: int
    stride: int

    def __post_init__(self):
        if self.length <= 0 or self.stride <= 0:
            raise ValueError("length and stride must be positive")

    def length_timedelta(self, freq_seconds: int) -> timedelta:
        return timedelta(seconds=self.length * freq_seconds)

    def stride_timedelta(self, freq_seconds: int) -> timedelta:
        return timedelta(seconds=self.stride * freq_seconds)


def generate_window_bounds(
    start: datetime, end: datetime, freq_seconds: int, spec: WindowSpec
) -> Iterator[tuple[int, datetime, datetime]]:
    """Chronological (window_index, window_start, window_end) triples, end exclusive.

    Windows are only ever a function of ``start`` and elapsed time -- generating them
    never looks past ``window_end``, so consuming this iterator in order is inherently
    causal (safe for a production/streaming simulation: never peek at future samples).
    """
    length = spec.length_timedelta(freq_seconds)
    step = spec.stride_timedelta(freq_seconds)
    i = 0
    cursor = start
    while cursor + length <= end:
        yield i, cursor, cursor + length
        cursor += step
        i += 1


def window_bounds_frame(start: datetime, end: datetime, freq_seconds: int, spec: WindowSpec) -> pl.DataFrame:
    rows = list(generate_window_bounds(start, end, freq_seconds, spec))
    return pl.DataFrame(
        rows, schema={"window_index": pl.Int64, "window_start": pl.Datetime("us"), "window_end": pl.Datetime("us")}, orient="row"
    )


def label_windows(
    labels_long: pl.DataFrame,
    bounds: pl.DataFrame,
    global_start: datetime,
    freq_seconds: int,
    spec: WindowSpec,
    channels: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Tag every window nominal/non-nominal without an O(n_windows * n_labels) scan.

    A single annotated timestamp can fall inside up to ``ceil(length / stride)``
    overlapping windows. Rather than filtering the (potentially multi-million-row)
    ``labels_long`` table once per window, each label row is matched against its small,
    bounded set of candidate window indices in one vectorized cross join.
    """
    if channels is not None:
        labels_long = labels_long.filter(pl.col("channel").is_in(channels))

    if labels_long.height == 0:
        return bounds.with_columns(
            pl.lit(True).alias("is_nominal"),
            pl.lit([], dtype=pl.List(pl.Int8)).alias("category_codes"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("channels_affected"),
        )

    stride_s = spec.stride * freq_seconds
    length_s = spec.length * freq_seconds
    k = math.ceil(spec.length / spec.stride)  # max windows a single timestamp can belong to

    offsets = pl.DataFrame({"offset": list(range(k))})
    matched = (
        labels_long.with_columns(
            ((pl.col("time") - global_start).dt.total_seconds() // stride_s).cast(pl.Int64).alias("base_index")
        )
        .join(offsets, how="cross")
        .with_columns((pl.col("base_index") - pl.col("offset")).alias("window_index"))
        .filter(pl.col("window_index") >= 0)
        .with_columns(
            (pl.lit(global_start) + pl.col("window_index") * pl.duration(seconds=stride_s)).alias("w_start")
        )
        .filter((pl.col("time") >= pl.col("w_start")) & (pl.col("time") < pl.col("w_start") + pl.duration(seconds=length_s)))
        .group_by("window_index")
        .agg(
            pl.col("category_code").unique().alias("category_codes"),
            pl.col("channel").unique().alias("channels_affected"),
        )
    )

    result = bounds.join(matched, on="window_index", how="left").with_columns(
        pl.col("category_codes").fill_null([]),
        pl.col("channels_affected").fill_null([]),
    )
    return result.with_columns((pl.col("category_codes").list.len() == 0).alias("is_nominal"))


def window_row_slices(n_rows: int, spec: WindowSpec) -> Iterator[tuple[int, int, int]]:
    """(window_index, row_start, row_end) triples over an already-materialized array.

    The resampled grid is uniform and gap-free by construction (verified against real
    Mission1/Mission2 output), so once a mission's values are loaded into a plain
    ``(n_rows, n_channels)`` array, row position and elapsed time are interchangeable --
    slicing by integer row offsets is equivalent to, and far cheaper than, repeatedly
    filtering a Parquet scan by timestamp per window (what `load_window_frame` does for a
    single one-off window). Index progression matches `generate_window_bounds` exactly
    (both start at offset 0 and step by ``spec.stride``), so `window_index` here lines up
    with the same window's entry in `label_windows`'s output.
    """
    i = 0
    start = 0
    while start + spec.length <= n_rows:
        yield i, start, start + spec.length
        start += spec.stride
        i += 1


def load_window_frame(chunks_glob: str | Path, window_start: datetime, window_end: datetime) -> pl.DataFrame:
    """Materialize a single window's multivariate rows (e.g. to feed a model)."""
    return (
        pl.scan_parquet(str(chunks_glob))
        .filter((pl.col("time") >= window_start) & (pl.col("time") < window_end))
        .collect()
    )
