from __future__ import annotations

from enum import Enum

import polars as pl

from esa_adb.config import MissionConfig


class ChannelType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CONSTANT = "constant"
    CATEGORICAL = "categorical"
    MONOTONIC = "monotonic"


_MONO_EPS = 1e-6


def infer_channel_types(
    train_lf: pl.LazyFrame,
    channel_meta: pl.DataFrame,
    config: MissionConfig,
    channels: list[str],
) -> pl.DataFrame:
    """Classify each channel for standardization purposes.

    Precedence: (1) group-metadata override -- if channels.csv places a channel in one
    of `config.monotonic_groups`, it's MONOTONIC per the paper's exact definition; (2)
    constant/binary, which are exact data facts that take priority over any label; (3) the
    real dataset's own `channels.csv` "Categorical" YES/NO flag (not mentioned in the
    paper text, but present in the actual Zenodo release, and more precise than guessing
    from cardinality alone); (4) a data-driven fallback otherwise.

    Approximation note: the data-driven stats below are computed over ALL train rows,
    filtered only by the wide `any_target_anomaly` convenience flag (rows where some
    *target* channel has an active anomaly are excluded) rather than an exact
    per-channel nominal mask. Anomalies are a small minority of rows, so this has
    negligible effect on a *typing* heuristic (unlike standardization stats, which are
    numerically sensitive and use exact per-channel exclusion -- see standardize.py).

    Stats are computed **one channel at a time** (like standardize.py), not as a single
    combined multi-column aggregation -- a mission with many channels and a long train
    span (e.g. Mission1: 76 channels x 7 years) makes a single mega-`.select()` over all
    channels at once materialize enough intermediate state (the `.diff()` monotonicity
    check in particular) to risk an OOM kill. Looping trades some repeated scanning of
    the underlying Parquet files for a peak memory footprint bounded by one channel's
    column, consistent with this pipeline's laptop-friendly design elsewhere.
    """
    if "any_target_anomaly" in train_lf.collect_schema().names():
        nominal_lf = train_lf.filter(~pl.col("any_target_anomaly"))
    else:
        nominal_lf = train_lf

    stats_row = {}
    for ch in channels:
        row = (
            nominal_lf.select(
                pl.col(ch).n_unique().alias("n_unique"),
                (pl.col(ch).cast(pl.Float64) - pl.col(ch).cast(pl.Float64).round(0))
                .abs()
                .max()
                .alias("frac_max"),
                (pl.col(ch).diff() >= -_MONO_EPS).all().alias("nondecreasing"),
                (pl.col(ch).diff() <= _MONO_EPS).all().alias("nonincreasing"),
            )
            .collect()
            .row(0, named=True)
        )
        stats_row[f"{ch}__n_unique"] = row["n_unique"]
        stats_row[f"{ch}__frac_max"] = row["frac_max"]
        stats_row[f"{ch}__nondecreasing"] = row["nondecreasing"]
        stats_row[f"{ch}__nonincreasing"] = row["nonincreasing"]

    group_by_channel = {}
    is_target_by_channel = {}
    is_categorical_by_channel = {}
    if "channel" in channel_meta.columns:
        for row in channel_meta.iter_rows(named=True):
            group_by_channel[row["channel"]] = row.get("group")
            is_target_by_channel[row["channel"]] = row.get("is_target")
            is_categorical_by_channel[row["channel"]] = row.get("is_categorical")

    rows = []
    for ch in channels:
        n_unique = stats_row[f"{ch}__n_unique"]
        frac_max = stats_row[f"{ch}__frac_max"]
        nondecreasing = stats_row[f"{ch}__nondecreasing"]
        nonincreasing = stats_row[f"{ch}__nonincreasing"]
        group = group_by_channel.get(ch)

        if group is not None and group in config.monotonic_groups:
            ch_type, typed_by = ChannelType.MONOTONIC, "group-metadata"
        elif n_unique is None or n_unique <= 1:
            ch_type, typed_by = ChannelType.CONSTANT, "data-driven"
        elif n_unique == 2:
            ch_type, typed_by = ChannelType.BINARY, "data-driven"
        elif is_categorical_by_channel.get(ch):
            ch_type, typed_by = ChannelType.CATEGORICAL, "channels.csv"
        elif n_unique <= config.categorical_max_unique and (frac_max is None or frac_max < 1e-6):
            ch_type, typed_by = ChannelType.CATEGORICAL, "data-driven"
        elif nondecreasing or nonincreasing:
            ch_type, typed_by = ChannelType.MONOTONIC, "data-driven"
        else:
            ch_type, typed_by = ChannelType.CONTINUOUS, "data-driven"

        rows.append(
            {
                "channel": ch,
                "type": ch_type.value,
                "n_unique_train": n_unique,
                "typed_by": typed_by,
                "group": group,
                "is_target": is_target_by_channel.get(ch),
            }
        )

    return pl.DataFrame(rows)
