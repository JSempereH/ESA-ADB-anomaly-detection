#!/usr/bin/env python3
"""Fit and evaluate the lightweight detectors against a mission's real processed data.

Usage: python scripts/demo_lightweight_models.py --mission 2 [--test-days N]

Evaluates over the *entire* test split by default (`--test-days` only exists to bound
runtime for quick iteration). Reports two views:
  - window-level precision/recall/F-beta (evaluate_window_predictions) -- an aggregate,
    strict metric: every window inside a long event that isn't individually flagged
    counts against recall.
  - per-event detection (evaluate_events / summarize_event_detection) -- did the model
    flag *any* row within each real annotated event's [StartTime, EndTime]? Broken down
    by event Category (anomaly / rare nominal event / ...).

Neither is the paper's corrected event-wise / affiliation-based / ADTQC metric suite --
see README's "Known approximations" section before comparing these numbers to the paper.
"""
import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.evaluation import evaluate_events, evaluate_window_predictions, summarize_event_detection
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.lightweight import GlobalSTD, WindowedIsolationForest
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--test-days", type=int, default=None, help="bound the test split (default: full test split)")
    parser.add_argument("--window-length", type=int, default=10, help="samples per window")
    parser.add_argument("--window-stride", type=int, default=10, help="samples between window starts")
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)
    channel_meta = load_channel_metadata(Path(config.raw_data_dir) / "channels.csv")
    target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()
    print(f"Mission{args.mission}: {len(target_channels)} target channels")

    chunks_glob = str(Path(config.processed_dir) / "chunks" / "*.parquet")
    lf = pl.scan_parquet(chunks_glob)

    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
    events = pl.read_csv(
        Path(config.interim_dir) / f"anomalies_mission{args.mission}.csv", try_parse_dates=True
    ).filter(pl.col("Channel").is_in(target_channels))

    train_raw = lf.filter(pl.col("split") == "train").select(["time", *target_channels]).collect()
    train_nominal = filter_nominal_rows(train_raw, labels_long, target_channels)
    print(f"Train nominal rows: {train_nominal.height} / {train_raw.height}")

    test_start = config.split.test_start
    test_end = config.split.test_end if args.test_days is None else min(config.split.test_end, test_start + timedelta(days=args.test_days))
    test_df = (
        lf.filter((pl.col("time") >= test_start) & (pl.col("time") < test_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    span_desc = "full test split" if args.test_days is None else f"{args.test_days}-day bounded window"
    print(f"Test rows ({span_desc}): {test_df.height}")

    stats = pl.read_parquet(Path(config.processed_dir) / "standardization_stats.parquet")
    global_std = GlobalSTD.from_standardization_stats(stats, target_channels, n_std=3.0)

    spec = WindowSpec(length=args.window_length, stride=args.window_stride)
    t0 = time.time()
    iforest = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(
        channel_matrix(train_nominal, target_channels)
    )
    print(f"WindowedIsolationForest fit in {time.time() - t0:.1f}s")

    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    labelled_bounds = label_windows(
        labels_long, bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels
    )
    print(f"Windows in test slice: {labelled_bounds.height} ({(~labelled_bounds['is_nominal']).sum()} non-nominal)")

    def report(name: str, row_pred, t_fit: float) -> None:
        window_pred = [bool(row_pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
        window_metrics = evaluate_window_predictions(window_pred, labelled_bounds, beta=0.5)
        event_report = evaluate_events(row_pred, test_times, events)
        event_summary = summarize_event_detection(event_report)
        print(f"\n=== {name} (scored in {t_fit:.1f}s) ===")
        print("Window-level:", window_metrics)
        print("Event-level detection rate:")
        print(event_summary)

    t0 = time.time()
    # WindowedIsolationForest.predict() is window-indexed (scikit-learn's own
    # contamination-based decision); broadcast to row-level so it's comparable to
    # GlobalSTD's row-level predictions and usable by evaluate_events (row-indexed).
    iforest_window_pred = iforest.predict(test_values)
    iforest_row_pred = broadcast_window_pred_to_rows(iforest_window_pred, test_values.shape[0], spec)
    report("WindowedIsolationForest", iforest_row_pred, time.time() - t0)

    t0 = time.time()
    global_std_row_pred = global_std.predict(test_values)
    report("GlobalSTD", global_std_row_pred, time.time() - t0)


if __name__ == "__main__":
    main()
