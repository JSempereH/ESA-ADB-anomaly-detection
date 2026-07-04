#!/usr/bin/env python3
"""Fit and evaluate PCAReconstructor (fixed vs. rolling-percentile threshold) against a
mission's real processed data, alongside WindowedIsolationForest for comparison.

Usage: python scripts/demo_pca_and_dynamic_threshold.py --mission 2 [--test-days N]

Runs over the entire test split by default. See demo_lightweight_models.py's docstring
for what the window-level vs. event-level reports mean, and README's "Known
approximations" section before comparing these numbers to the paper.
"""
import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.evaluation import evaluate_events, evaluate_window_predictions, summarize_event_detection
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.dynamic_threshold import RollingPercentileThreshold
from esa_adb.models.lightweight import PCAReconstructor, WindowedIsolationForest
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--test-days", type=int, default=None, help="bound the test split (default: full test split)")
    parser.add_argument("--val-days", type=int, default=30, help="bound the validation split used for fixed-threshold calibration")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="fraction of variance PCA components must explain")
    parser.add_argument("--rolling-window", type=int, default=20_000, help="rolling threshold window size, in samples")
    parser.add_argument(
        "--rolling-recompute-every",
        type=int,
        default=500,
        help="only re-sort the rolling window every N updates (1 = exact but O(n log n) per sample -- "
        "catastrophic over millions of rows; see RollingPercentileThreshold's docstring)",
    )
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

    val_end = config.split.train_end
    val_start = max(config.split.val_start, val_end - timedelta(days=args.val_days))
    val_raw = lf.filter((pl.col("time") >= val_start) & (pl.col("time") < val_end)).select(["time", *target_channels]).collect()
    val_nominal = filter_nominal_rows(val_raw, labels_long, target_channels)
    print(f"Validation nominal rows ({args.val_days}-day bounded window): {val_nominal.height} / {val_raw.height}")

    test_start = config.split.test_start
    test_end = config.split.test_end if args.test_days is None else min(config.split.test_end, test_start + timedelta(days=args.test_days))
    test_df = lf.filter((pl.col("time") >= test_start) & (pl.col("time") < test_end)).select(["time", *target_channels]).sort("time").collect()
    span_desc = "full test split" if args.test_days is None else f"{args.test_days}-day bounded window"
    print(f"Test rows ({span_desc}): {test_df.height}")

    train_values = channel_matrix(train_nominal, target_channels)
    val_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    t0 = time.time()
    pca = PCAReconstructor(n_components=args.pca_variance).fit(train_values)
    pca.calibrate_threshold(val_values, percentile=99.0)
    print(
        f"PCAReconstructor fit in {time.time() - t0:.1f}s: {pca.pca.n_components_} components "
        f"explain {args.pca_variance:.0%} variance (of {len(target_channels)} channels), threshold={pca.threshold_:.4f}"
    )

    spec = WindowSpec(length=10, stride=10)
    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    labelled_bounds = label_windows(labels_long, bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)
    print(f"Windows in test slice: {labelled_bounds.height} ({(~labelled_bounds['is_nominal']).sum()} non-nominal)")

    def report(name: str, row_pred: np.ndarray, t_elapsed: float) -> None:
        window_pred = [bool(row_pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
        window_metrics = evaluate_window_predictions(window_pred, labelled_bounds, beta=0.5)
        event_summary = summarize_event_detection(evaluate_events(row_pred, test_times, events))
        print(f"\n=== {name} (scored in {t_elapsed:.1f}s) ===")
        print("Window-level:", window_metrics)
        print("Event-level detection rate:")
        print(event_summary)

    t0 = time.time()
    pca_scores = pca.score(test_values)
    fixed_pred = pca_scores > pca.threshold_
    report("PCAReconstructor (fixed threshold)", fixed_pred, time.time() - t0)

    t0 = time.time()
    rolling = RollingPercentileThreshold(
        window_size=args.rolling_window,
        percentile=99.0,
        initial_scores=pca.score(val_values),
        recompute_every=args.rolling_recompute_every,
    )
    dynamic_pred = rolling.predict_stream(pca_scores)
    report(f"PCAReconstructor (rolling threshold, window={args.rolling_window})", dynamic_pred, time.time() - t0)

    t0 = time.time()
    iforest = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(train_values)
    iforest_window_pred = iforest.predict(test_values)
    iforest_row_pred = broadcast_window_pred_to_rows(iforest_window_pred, test_values.shape[0], spec)
    report("WindowedIsolationForest (for comparison)", iforest_row_pred, time.time() - t0)


if __name__ == "__main__":
    main()
