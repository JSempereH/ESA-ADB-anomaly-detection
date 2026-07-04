from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import polars as pl


@dataclass
class CarryState:
    time: Optional[datetime] = None
    value: Optional[float] = None
    label_code: Optional[int] = None


def _prepend_carry(raw: pl.DataFrame, carry: Optional[CarryState]) -> pl.DataFrame:
    if carry is None or carry.time is None:
        return raw
    synthetic = pl.DataFrame(
        {"time": [carry.time], "value": [carry.value], "label_code": [carry.label_code]}
    ).with_columns(
        pl.col("value").cast(raw.schema["value"]),
        pl.col("label_code").cast(raw.schema["label_code"]),
    )
    return pl.concat([synthetic, raw], how="vertical").sort("time")


def zero_order_hold_resample(
    channel_chunk_raw: pl.DataFrame, grid: pl.DataFrame, carry: Optional[CarryState]
) -> tuple[pl.DataFrame, CarryState]:
    """Forward-fill (zero-order hold) a channel's raw samples onto a fixed grid.

    ``channel_chunk_raw`` must already be limited to this chunk's time window; the
    previous chunk's last known sample is threaded in via ``carry`` so resampling is
    correct across chunk boundaries without ever holding more than one chunk in memory.
    Any grid points still null after the asof join can only be a genuine leading gap
    (before the channel's very first raw sample overall); they're filled with that first
    known sample -- this is the paper's accepted "small future information leak", limited
    to the first few samples of a channel's timeline.
    """
    raw = _prepend_carry(channel_chunk_raw, carry)

    if raw.height == 0:
        resampled = grid.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("value"),
            pl.lit(None, dtype=pl.Int8).alias("label_code"),
        )
    else:
        raw_sorted = raw.sort("time")
        resampled = grid.sort("time").join_asof(raw_sorted, on="time", strategy="backward")

        # Any remaining null can only be a leading gap (before the channel's very first
        # raw sample overall -- asof-backward guarantees non-null for every grid point at
        # or after that). Fill it with the literal first known raw sample, per the paper
        # -- NOT a plain backward-fill over the resampled column, which would instead pull
        # in whatever a *later* raw sample already produced for the next grid tick.
        first_row = raw_sorted.row(0, named=True)
        resampled = resampled.with_columns(
            pl.col("value").fill_null(pl.lit(first_row["value"])),
            pl.col("label_code").fill_null(pl.lit(first_row["label_code"])),
        )

    if channel_chunk_raw.height > 0:
        last = channel_chunk_raw.sort("time").tail(1).row(0, named=True)
        new_carry = CarryState(time=last["time"], value=last["value"], label_code=last["label_code"])
    else:
        new_carry = carry if carry is not None else CarryState()

    return resampled, new_carry


def correct_point_anomalies(resampled: pl.DataFrame, channel_chunk_raw: pl.DataFrame) -> pl.DataFrame:
    """Rescue annotated raw samples shadowed by a later nominal sample in the same bin.

    For each pair of consecutive, currently-nominal grid points (g_i, g_{i+1}), if an
    annotated raw sample falls strictly inside (g_i, g_{i+1}), it would otherwise be lost
    to the zero-order-hold join (which only keeps the *last* raw sample per bin). That
    sample's value/label is reassigned to g_{i+1}, per the paper's correction rule.
    """
    annotated_raw = channel_chunk_raw.filter(pl.col("label_code") != 0)
    if annotated_raw.height == 0:
        return resampled

    resampled = resampled.with_columns(
        pl.col("time").shift(-1).alias("next_time"),
        pl.col("label_code").shift(-1).alias("next_label"),
    )
    candidates = resampled.filter(
        (pl.col("label_code") == 0) & (pl.col("next_label") == 0) & pl.col("next_time").is_not_null()
    )
    if candidates.height == 0:
        return resampled.drop(["next_time", "next_label"])

    annotated_sorted = annotated_raw.sort("time").rename(
        {"time": "raw_time", "value": "raw_value", "label_code": "raw_label"}
    )
    matched = (
        candidates.select("time", "next_time")
        .sort("next_time")
        .join_asof(annotated_sorted, left_on="next_time", right_on="raw_time", strategy="backward")
        .filter(pl.col("raw_time").is_not_null() & (pl.col("raw_time") > pl.col("time")))
        .select(
            pl.col("next_time").alias("time"),
            pl.col("raw_value").alias("corrected_value"),
            pl.col("raw_label").alias("corrected_label"),
        )
    )

    resampled = resampled.join(matched, on="time", how="left").with_columns(
        pl.coalesce(["corrected_value", "value"]).alias("value"),
        pl.coalesce(["corrected_label", "label_code"]).alias("label_code"),
    )
    return resampled.drop(["next_time", "next_label", "corrected_value", "corrected_label"])


def resample_channel_chunk(
    channel_chunk_raw: pl.DataFrame,
    grid: pl.DataFrame,
    carry: Optional[CarryState],
    channel_name: str,
) -> tuple[pl.DataFrame, CarryState]:
    resampled, new_carry = zero_order_hold_resample(channel_chunk_raw, grid, carry)
    resampled = correct_point_anomalies(resampled, channel_chunk_raw)
    resampled = resampled.rename({"value": channel_name, "label_code": f"{channel_name}__label"})
    return resampled, new_carry
