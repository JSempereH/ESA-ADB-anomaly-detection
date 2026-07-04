from __future__ import annotations

import numpy as np
import polars as pl


def print_time_difference(df: pl.DataFrame, time_col: str = "time") -> None:
    if time_col not in df.columns:
        print(f"No '{time_col}' column present.")
        return
    times = np.sort(df[time_col].to_numpy())
    diffs = np.diff(times)
    if len(diffs) == 0:
        print("Not enough time data to compute differences.")
        return
    print(f"Rows: {df.shape[0]}")
    print(f"Min/Max/Mean time diff: {diffs.min()} / {diffs.max()} / {diffs.mean()}")
