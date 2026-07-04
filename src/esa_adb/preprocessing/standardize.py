from __future__ import annotations

import polars as pl

from esa_adb.preprocessing.typing_ import ChannelType


def compute_standardization_stats(
    train_lf: pl.LazyFrame,
    labels_long: pl.DataFrame,
    channel_types: pl.DataFrame,
    channels: list[str],
) -> tuple[pl.DataFrame, dict[str, list]]:
    """Per-channel standardization parameters computed from train-nominal points only.

    Nominal exclusion here is exact and per-channel (via `labels_long`), unlike the
    approximation used in typing_.infer_channel_types -- these numbers feed directly
    into model inputs, so precision matters more here than for the type heuristic.
    """
    type_by_channel = dict(zip(channel_types["channel"], channel_types["type"]))
    rows = []
    categorical_mappings: dict[str, list] = {}

    for ch in channels:
        ch_type = type_by_channel.get(ch, ChannelType.CONTINUOUS.value)
        excluded_times = labels_long.filter(
            (pl.col("channel") == ch) & (pl.col("category_code") != 0)
        )["time"]

        col_lf = train_lf.select("time", ch).sort("time")
        if excluded_times.len() > 0:
            col_lf = col_lf.filter(~pl.col("time").is_in(excluded_times.to_list()))

        if ch_type == ChannelType.CATEGORICAL.value:
            ordered_values = col_lf.select(ch).collect()[ch].unique(maintain_order=True).to_list()
            categorical_mappings[ch] = ordered_values
            code_map = {v: i for i, v in enumerate(ordered_values)}
            col_lf = col_lf.with_columns(
                pl.col(ch).replace_strict(code_map, default=len(ordered_values), return_dtype=pl.Float64).alias(ch)
            )
        elif ch_type == ChannelType.MONOTONIC.value:
            col_lf = col_lf.with_columns(pl.col(ch).diff().alias(ch))

        agg = col_lf.select(
            pl.col(ch).mean().alias("mean"),
            pl.col(ch).std().alias("std"),
            pl.col(ch).min().alias("min"),
            pl.col(ch).max().alias("max"),
        ).collect()
        stat = agg.row(0, named=True)
        mean, std, cmin, cmax = stat["mean"], stat["std"], stat["min"], stat["max"]

        if ch_type == ChannelType.CONSTANT.value:
            transform = "mean_subtract"
        elif ch_type == ChannelType.BINARY.value:
            transform = "minmax"
        elif ch_type == ChannelType.MONOTONIC.value:
            transform = "diff_zscore"
        elif ch_type == ChannelType.CATEGORICAL.value:
            transform = "categorical_zscore"
        else:
            transform = "zscore"

        rows.append(
            {
                "channel": ch,
                "type": ch_type,
                "transform": transform,
                "mean": mean,
                "std": std if std and std > 0 else 1.0,
                "min": cmin,
                "max": cmax,
            }
        )

    return pl.DataFrame(rows), categorical_mappings


def apply_standardization(
    df: pl.DataFrame, stats: pl.DataFrame, categorical_mappings: dict[str, list]
) -> pl.DataFrame:
    for row in stats.iter_rows(named=True):
        ch, transform = row["channel"], row["transform"]
        if ch not in df.columns:
            continue

        if transform == "zscore":
            df = df.with_columns(((pl.col(ch) - row["mean"]) / row["std"]).alias(ch))
        elif transform == "mean_subtract":
            df = df.with_columns((pl.col(ch) - row["mean"]).alias(ch))
        elif transform == "minmax":
            span = row["max"] - row["min"] if row["max"] is not None and row["min"] is not None else None
            if span and span > 0:
                df = df.with_columns(((pl.col(ch) - row["min"]) / span).alias(ch))
        elif transform == "diff_zscore":
            df = df.with_columns(pl.col(ch).diff().alias(ch))
            df = df.with_columns(((pl.col(ch) - row["mean"]) / row["std"]).alias(ch))
        elif transform == "categorical_zscore":
            mapping = categorical_mappings.get(ch, [])
            code_map = {v: i for i, v in enumerate(mapping)}
            df = df.with_columns(
                pl.col(ch).replace_strict(code_map, default=len(mapping), return_dtype=pl.Float64).alias(ch)
            )
            df = df.with_columns(((pl.col(ch) - row["mean"]) / row["std"]).alias(ch))
    return df
