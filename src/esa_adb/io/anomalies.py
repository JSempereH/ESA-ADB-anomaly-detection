from __future__ import annotations

from pathlib import Path

import polars as pl

from esa_adb.preprocessing.labeling import CategoryCode, category_to_code


def build_anomalies_table(
    anomaly_types_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> pl.DataFrame:
    """Join labels.csv (per-segment annotations) with anomaly_types.csv (per-ID metadata).

    Keeps all event categories present in the dataset (anomaly, rare nominal event,
    communication gap, invalid segment). Extra descriptive columns from
    anomaly_types.csv (e.g. Class/Subclass, or Dimensionality/Locality/Length) are kept
    as-is via the join rather than hardcoding a column allowlist, so this doesn't
    silently drop metadata if the real schema differs slightly from what's assumed here.
    """
    anomaly_types = pl.read_csv(str(anomaly_types_path))
    labels = pl.read_csv(str(labels_path), try_parse_dates=True)

    if labels["StartTime"].dtype != pl.Datetime:
        labels = labels.with_columns(pl.col("StartTime").str.to_datetime(strict=False))
    if labels["EndTime"].dtype != pl.Datetime:
        labels = labels.with_columns(pl.col("EndTime").str.to_datetime(strict=False))

    # The real ESA-AD release parses StartTime/EndTime as UTC-aware, but the raw channel
    # pickles carry timezone-naive timestamps - normalize both to naive microsecond
    # datetimes so later asof-joins against per-channel raw data (attach_raw_labels) have
    # matching key dtypes instead of erroring or silently mismatching.
    for col in ("StartTime", "EndTime"):
        dtype = labels.schema[col]
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            labels = labels.with_columns(pl.col(col).dt.replace_time_zone(None))
        labels = labels.with_columns(pl.col(col).cast(pl.Datetime("us")))

    merged = labels.join(anomaly_types, on="ID", how="inner")
    merged = merged.with_columns(
        (pl.col("EndTime") - pl.col("StartTime")).alias("duration"),
        pl.col("Category").str.strip_chars().str.to_lowercase().alias("Category"),
    )
    merged = merged.with_columns(
        pl.col("Category")
        .map_elements(category_to_code, return_dtype=pl.Int8)
        .alias("category_code")
    )
    merged = merged.sort(["Channel", "StartTime"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.with_columns(pl.col("duration").dt.total_seconds().alias("duration_seconds")).drop(
        "duration"
    ).write_csv(str(output_path))
    return merged


def attach_raw_labels(
    channel_raw_df: pl.DataFrame, anomalies_df: pl.DataFrame, channel_name: str
) -> pl.DataFrame:
    """Per-channel interval join: attach a `label_code` to each raw sample of one channel.

    Only intervals annotated for this specific `channel_name` are considered, so an
    anomaly window never gets applied to a channel other than the one it was actually
    annotated for.
    """
    intervals = (
        anomalies_df.filter(pl.col("Channel") == channel_name)
        .select("StartTime", "EndTime", "category_code")
        .sort("StartTime")
    )

    if intervals.height == 0:
        return channel_raw_df.with_columns(pl.lit(int(CategoryCode.NOMINAL)).alias("label_code").cast(pl.Int8))

    joined = channel_raw_df.sort("time").join_asof(
        intervals, left_on="time", right_on="StartTime", strategy="backward"
    )
    joined = joined.with_columns(
        pl.when(
            pl.col("StartTime").is_not_null() & (pl.col("time") <= pl.col("EndTime"))
        )
        .then(pl.col("category_code"))
        .otherwise(int(CategoryCode.NOMINAL))
        .fill_null(int(CategoryCode.NOMINAL))
        .cast(pl.Int8)
        .alias("label_code")
    )
    return joined.drop(["StartTime", "EndTime", "category_code"])
