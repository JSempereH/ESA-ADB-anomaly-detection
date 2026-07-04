from __future__ import annotations

from datetime import datetime

from dateutil.relativedelta import relativedelta

from esa_adb.config import SplitBoundaries


def compute_split_boundaries(
    global_start: datetime, global_end: datetime, val_months: int = 3
) -> SplitBoundaries:
    """First half of the range = train, second half = test.

    The last ``val_months`` calendar months of the train half are carved out as
    validation. Matches the ESA-ADB paper's split: chronological, no shuffling.
    """
    midpoint = global_start + (global_end - global_start) / 2
    train_end = midpoint
    test_start = midpoint
    val_start = train_end - relativedelta(months=val_months)

    boundaries = SplitBoundaries(
        train_start=global_start,
        val_start=val_start,
        train_end=train_end,
        test_start=test_start,
        test_end=global_end,
    )
    _validate(boundaries)
    return boundaries


def _validate(b: SplitBoundaries) -> None:
    if not (b.train_start <= b.val_start <= b.train_end <= b.test_start <= b.test_end):
        raise ValueError(
            "Split boundaries must be non-decreasing and fully partition the date range: "
            f"{b.train_start} <= {b.val_start} <= {b.train_end} <= {b.test_start} <= {b.test_end}"
        )


def label_split(time_col: str, boundaries: SplitBoundaries):
    """Return a Polars expression labelling each row 'train' / 'val' / 'test'."""
    import polars as pl

    return (
        pl.when(pl.col(time_col) < boundaries.val_start)
        .then(pl.lit("train"))
        .when(pl.col(time_col) < boundaries.test_start)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
