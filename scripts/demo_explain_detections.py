#!/usr/bin/env python3
"""Aggregates `most_anomalous_channels()` over every real, correctly-detected event in
Mission2's test period (using `Autoencoder`, this project's best single-split model) to
answer an actual question about the dataset: is there a small set of "usual suspect"
channels/subsystems behind most detections, or is it spread out?

For each real annotated event `Autoencoder` detects (>=1 overlapping flagged row), takes
the single highest-scoring row within that event's span (the "peak" of the detection)
and asks which channel(s) drove that peak -- one clean answer per real event, rather than
a per-row histogram dominated by long detections repeating the same answer many times.

Usage: python scripts/demo_explain_detections.py --mission 2 [--top-n 3]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import channel_matrix
from esa_adb.models.heavy import Autoencoder
from esa_adb.preprocessing.labeling import filter_nominal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)
    channel_meta = load_channel_metadata(Path(config.raw_data_dir) / "channels.csv")
    target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()
    channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta["subsystem"])) if "subsystem" in channel_meta.columns else {}
    print(f"Mission{args.mission}: {len(target_channels)} target channels")

    chunks_glob = str(Path(config.processed_dir) / "chunks" / "*.parquet")
    lf = pl.scan_parquet(chunks_glob)
    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
    events = pl.read_csv(Path(config.interim_dir) / f"anomalies_mission{args.mission}.csv", try_parse_dates=True).filter(
        pl.col("Channel").is_in(target_channels)
    )

    train_raw = lf.filter(pl.col("split") == "train").select(["time", *target_channels]).collect()
    train_nominal = filter_nominal_rows(train_raw, labels_long, target_channels)
    val_df = (
        lf.filter((pl.col("time") >= config.split.val_start) & (pl.col("time") < config.split.train_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    val_nominal = filter_nominal_rows(val_df, labels_long, target_channels)
    test_df = (
        lf.filter((pl.col("time") >= config.split.test_start) & (pl.col("time") < config.split.test_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )

    train_values = channel_matrix(train_nominal, target_channels)
    val_nominal_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    print(f"Training Autoencoder on {train_values.shape[0]} rows for {args.ae_epochs} epochs...")
    model = Autoencoder(n_channels=len(target_channels), hidden_sizes=(32, 16), latent_size=8)
    model.fit(train_values, epochs=args.ae_epochs, batch_size=2048, verbose=False)
    model.calibrate_threshold(val_nominal_values, percentile=99.0)

    scores = model.score(test_values)
    row_pred = scores > model.threshold_
    print(f"Test rows: {test_values.shape[0]}, flagged: {int(row_pred.sum())}")

    eval_start, eval_end = test_times.min(), test_times.max()
    in_range_events = events.filter((pl.col("EndTime") >= eval_start) & (pl.col("StartTime") <= eval_end))
    print(f"Annotated events in test period: {in_range_events.height}")

    channel_counter: Counter = Counter()
    subsystem_counter: Counter = Counter()
    per_category: dict[str, Counter] = {}
    n_detected = 0

    for row in in_range_events.iter_rows(named=True):
        start, end = np.datetime64(row["StartTime"]), np.datetime64(row["EndTime"])
        mask = (test_times >= start) & (test_times <= end)
        if not mask.any() or not row_pred[mask].any():
            continue  # not detected -- nothing to explain
        n_detected += 1

        idx = np.where(mask)[0]
        local_scores = scores[idx]
        peak_row = test_values[idx[np.argmax(local_scores)]][None, :]
        top_channels = model.most_anomalous_channels(peak_row, channel_names=target_channels, top_n=args.top_n)[0]

        channel_counter.update(top_channels)
        subsystem_counter.update(channel_to_subsystem.get(c, "unknown") for c in top_channels)
        category = row.get("Category", "unknown")
        per_category.setdefault(category, Counter()).update(top_channels[:1])  # only the #1 channel, per category

    print(f"\nDetected events explained: {n_detected}")
    print(f"\nTop channels appearing in any event's top-{args.top_n} (all detected events):")
    for channel, count in channel_counter.most_common(15):
        print(f"  {channel}: {count} ({100 * count / n_detected:.1f}% of detected events)")

    if channel_to_subsystem:
        print("\nTop subsystems:")
        for subsystem, count in subsystem_counter.most_common():
            print(f"  {subsystem}: {count} ({100 * count / n_detected:.1f}% of detected events)")

    print("\nTop #1 channel, broken down by event category:")
    for category, counter in per_category.items():
        top = counter.most_common(3)
        print(f"  {category} ({sum(counter.values())} events): {top}")


if __name__ == "__main__":
    main()
