#!/usr/bin/env python3
"""Fit and evaluate the LSTM forecaster against a mission's real processed data.

Usage: python scripts/demo_heavy_model.py --mission 2 [--train-days 10] [--test-days N]

Training stays bounded by `--train-days`/`--val-days` (small by default -- this is a
CPU smoke test, not a real training run; see README). Scoring, however, runs over the
*entire* test split by default (`--test-days` only exists to bound it for quick
iteration) -- `LSTMForecaster.score()` processes long inputs in row-chunks, so this
doesn't require materializing every sliding window in memory at once.

Reports both window-level precision/recall/F-beta and per-event detection (broken down
by event Category) -- see demo_lightweight_models.py's docstring for what each means, and
README's "Known approximations" section before comparing these numbers to the paper.
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
from esa_adb.models.base import channel_matrix
from esa_adb.models.heavy import LSTMForecaster
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--train-days", type=int, default=10, help="bound the train split for a fast demo run")
    parser.add_argument("--val-days", type=int, default=10, help="bound the validation split used for threshold calibration")
    parser.add_argument("--test-days", type=int, default=None, help="bound the test split (default: full test split)")
    parser.add_argument("--input-length", type=int, default=50, help="LSTM context length, in samples")
    parser.add_argument("--epochs", type=int, default=5)
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

    train_start = config.split.train_start
    train_end = min(config.split.val_start, train_start + timedelta(days=args.train_days))
    train_raw = (
        lf.filter((pl.col("time") >= train_start) & (pl.col("time") < train_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    train_nominal = filter_nominal_rows(train_raw, labels_long, target_channels)
    print(f"Train nominal rows ({args.train_days}-day bounded window): {train_nominal.height} / {train_raw.height}")

    # Validation is the tail end of the val split, immediately adjacent to test_start --
    # closer in time to the test period than the start of a 3-month val window would be,
    # so its residual distribution is a more representative threshold-calibration target.
    val_end = config.split.train_end
    val_start = max(config.split.val_start, val_end - timedelta(days=args.val_days))
    val_raw = (
        lf.filter((pl.col("time") >= val_start) & (pl.col("time") < val_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    val_nominal = filter_nominal_rows(val_raw, labels_long, target_channels)
    print(f"Validation nominal rows ({args.val_days}-day bounded window): {val_nominal.height} / {val_raw.height}")

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

    train_values = channel_matrix(train_nominal, target_channels)
    val_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    model = LSTMForecaster(n_channels=len(target_channels), input_length=args.input_length, hidden_size=32, num_layers=1)
    t0 = time.time()
    model.fit(train_values, validation_values=val_values, epochs=args.epochs, batch_size=128, verbose=True)
    print(f"LSTMForecaster fit in {time.time() - t0:.1f}s, residual_std={model.residual_std_:.4f}, threshold={model.threshold_:.4f}")

    t0 = time.time()
    row_pred = model.predict(test_values)
    print(f"Scored full test split in {time.time() - t0:.1f}s. Rows flagged: {row_pred.sum()} / {len(row_pred)}")

    spec = WindowSpec(length=10, stride=10)
    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    labelled_bounds = label_windows(
        labels_long, bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels
    )
    print(f"Windows in test slice: {labelled_bounds.height} ({(~labelled_bounds['is_nominal']).sum()} non-nominal)")

    window_pred = [bool(row_pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
    window_metrics = evaluate_window_predictions(window_pred, labelled_bounds, beta=0.5)
    event_report = evaluate_events(row_pred, test_times, events)
    event_summary = summarize_event_detection(event_report)
    print("\n=== LSTMForecaster ===")
    print("Window-level:", window_metrics)
    print("Event-level detection rate:")
    print(event_summary)


if __name__ == "__main__":
    main()
