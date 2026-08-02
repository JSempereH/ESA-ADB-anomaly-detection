#!/usr/bin/env python3
"""Build grid-aligned telecommand features for a mission -- the P4 wiring of
`preprocessing/telecommands.py`'s so-far-unused encoding into something models consume.

Rather than the paper's 1-binary-column-per-telecommand encoding (123 extra mostly-zero
channels for Mission2, tripling input width), this produces three compact columns on
the SAME uniform grid as `chunks/*.parquet`:

- `tc_count`:   telecommand executions falling in this grid bin (all TCs pooled)
- `tc_any`:     1 if any TC fired in this bin
- `tc_recency`: exp(-seconds_since_last_TC / 1800) -- decays from 1 to 0 with a 30-min
                half-life-ish scale, so models can learn "the satellite was commanded
                recently, expect transients" instead of flagging every commanded
                maneuver as an anomaly.

Output: data/processed/Mission{N}/telecommands.parquet (time + 3 feature columns).
Consumed by `run_benchmark.py --with-telecommands` (extra model INPUT channels; never
targets, never part of the labeled channel set).

Usage: python scripts/build_telecommand_features.py --mission 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import polars as pl

from esa_adb.config import load_mission_config

RECENCY_SCALE_SECONDS = 1800.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()

    config = load_mission_config(Path(args.config_dir) / f"mission{args.mission}.yaml")
    tc_dir = Path(config.raw_data_dir) / "telecommands"
    tc_files = sorted(tc_dir.iterdir())
    print(f"{len(tc_files)} telecommand files in {tc_dir}")

    all_times = []
    for f in tc_files:
        df = pd.read_pickle(f)
        all_times.append(pl.Series("time", pd.to_datetime(df.index)).cast(pl.Datetime("us")))
    tc_times = pl.DataFrame({"time": pl.concat(all_times)})
    print(f"{tc_times.height} total telecommand executions")

    freq = f"{config.target_frequency_seconds}s"
    counts = (
        tc_times.with_columns(pl.col("time").dt.truncate(freq))
        .group_by("time")
        .len()
        .rename({"len": "tc_count"})
        .sort("time")
    )

    grid = (
        pl.scan_parquet(str(Path(config.processed_dir) / "chunks" / "*.parquet"))
        .select("time")
        .sort("time")
        .collect()
    )
    features = grid.join(counts, on="time", how="left").with_columns(pl.col("tc_count").fill_null(0).cast(pl.UInt16))
    features = features.with_columns((pl.col("tc_count") > 0).cast(pl.Int8).alias("tc_any"))

    # Recency: forward-fill the time of the last bin containing a TC, then decay.
    features = features.with_columns(
        pl.when(pl.col("tc_count") > 0).then(pl.col("time")).otherwise(None).alias("last_tc").forward_fill()
    )
    seconds_since = (features["time"] - features["last_tc"]).dt.total_seconds().fill_null(np.inf).to_numpy()
    features = features.with_columns(pl.Series("tc_recency", np.exp(-seconds_since / RECENCY_SCALE_SECONDS)).cast(pl.Float32))
    features = features.drop("last_tc")

    out_path = Path(config.processed_dir) / "telecommands.parquet"
    features.write_parquet(out_path)
    in_command = features.filter(pl.col("tc_any") == 1).height
    print(f"wrote {out_path}: {features.height} rows, {in_command} bins with >=1 TC ({100 * in_command / features.height:.2f}%)")


if __name__ == "__main__":
    main()
