from __future__ import annotations

from enum import IntEnum

import numpy as np
import polars as pl


class CategoryCode(IntEnum):
    NOMINAL = 0
    ANOMALY = 1
    RARE_NOMINAL_EVENT = 2
    COMMUNICATION_GAP = 3
    INVALID_SEGMENT = 4


_CATEGORY_TO_CODE = {
    "anomaly": CategoryCode.ANOMALY,
    "rare nominal event": CategoryCode.RARE_NOMINAL_EVENT,
    "rare event": CategoryCode.RARE_NOMINAL_EVENT,  # observed alias in some ESA-AD releases
    "communication gap": CategoryCode.COMMUNICATION_GAP,
    "invalid segment": CategoryCode.INVALID_SEGMENT,
}


def category_to_code(category: str) -> int:
    key = category.strip().lower()
    if key not in _CATEGORY_TO_CODE:
        raise ValueError(f"Unknown anomaly category: {category!r}")
    return int(_CATEGORY_TO_CODE[key])


def extract_labels_long(resampled_channel_df: pl.DataFrame, channel: str) -> pl.DataFrame:
    """Sparse (time, channel, category_code) rows for non-nominal grid points only."""
    return (
        resampled_channel_df.filter(pl.col("label_code") != int(CategoryCode.NOMINAL))
        .select(
            pl.col("time"),
            pl.lit(channel).alias("channel"),
            pl.col("label_code").alias("category_code"),
        )
    )


def any_target_anomaly_expr(channel_label_cols: list[str]) -> pl.Expr:
    """OR of `label_code == ANOMALY` across the given (target) channel label columns."""
    if not channel_label_cols:
        return pl.lit(False).alias("any_target_anomaly")
    expr = pl.lit(False)
    for col in channel_label_cols:
        expr = expr | (pl.col(col) == int(CategoryCode.ANOMALY))
    return expr.alias("any_target_anomaly")


def filter_nominal_rows(df: pl.DataFrame, labels_long: pl.DataFrame, channels: list[str]) -> pl.DataFrame:
    """Rows where none of `channels` has ANY non-nominal label (all 4 categories).

    Deliberately stricter than the wide `any_target_anomaly` convenience column, which
    only flags the ANOMALY category -- rare nominal events, communication gaps and
    invalid segments all pass through it uncaught. For "train on nominal only"
    semi-supervised models this matters: training on rows `label_windows` would later
    count as ground-truth-positive (e.g. rare events) both contaminates the fit and
    biases threshold calibration low, since the model partly learns to predict those
    events as normal. Uses `labels_long`, so it doesn't require re-running preprocessing.
    """
    non_nominal_times = labels_long.filter(pl.col("channel").is_in(channels))["time"].unique().to_list()
    return df.filter(~pl.col("time").is_in(non_nominal_times))


def row_is_non_nominal(df: pl.DataFrame, labels_long: pl.DataFrame, channels: list[str]) -> np.ndarray:
    """Row-level complement of `filter_nominal_rows`: a boolean array aligned to `df`'s
    row order (True = at least one of `channels` has a non-nominal label at that exact
    timestamp), instead of dropping those rows. Used to build labeled ground truth for
    validation-based threshold tuning (`evaluation.find_best_threshold`) against
    row-level detectors (`PCAReconstructor`, `Autoencoder`), which don't have a natural
    window index the way `preprocessing.windowing.label_windows` assumes.
    """
    non_nominal_times = labels_long.filter(pl.col("channel").is_in(channels))["time"].unique().to_list()
    return df.select(pl.col("time").is_in(non_nominal_times).alias("flag"))["flag"].to_numpy()
