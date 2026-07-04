#!/usr/bin/env python3
"""Comprehensive comparison of all lightweight/heavy models against a mission's real
processed data, using validation-set threshold tuning (maximizing F0.5 via
evaluation.find_best_threshold) instead of each model's default percentile-based
calibration -- a direct, non-hyped answer to "how do we improve F0.5".

Usage: python scripts/demo_full_comparison.py --mission 2 [--test-days N] [--val-days N]

Compares:
  - WindowedIsolationForest, raw flattened window features (the existing baseline)
  - WindowedIsolationForest, statistical+spectral window features (models.window_features)
  - PCAReconstructor (linear reconstruction)
  - Autoencoder (nonlinear reconstruction, heavy)
all with thresholds tuned on labeled validation data instead of an arbitrary fixed
percentile. See demo_lightweight_models.py's docstring for what the window-level vs.
event-level reports mean, and README's "Known approximations" before comparing these
numbers to the paper.
"""
import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.evaluation import evaluate_events, evaluate_window_predictions, find_best_threshold, summarize_event_detection
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.heavy import Autoencoder
from esa_adb.models.lightweight import PCAReconstructor, WindowedIsolationForest
from esa_adb.models.window_features import extract_window_features
from esa_adb.preprocessing.labeling import filter_nominal_rows, row_is_non_nominal
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--test-days", type=int, default=None, help="bound the test split (default: full)")
    parser.add_argument("--val-days", type=int, default=None, help="bound the validation split (default: full)")
    parser.add_argument("--ae-epochs", type=int, default=10)
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

    val_start = config.split.val_start
    val_end = config.split.train_end if args.val_days is None else min(config.split.train_end, val_start + timedelta(days=args.val_days))
    val_df = lf.filter((pl.col("time") >= val_start) & (pl.col("time") < val_end)).select(["time", *target_channels]).sort("time").collect()
    print(f"Validation rows (labeled, for threshold tuning): {val_df.height}")

    test_start = config.split.test_start
    test_end = config.split.test_end if args.test_days is None else min(config.split.test_end, test_start + timedelta(days=args.test_days))
    test_df = lf.filter((pl.col("time") >= test_start) & (pl.col("time") < test_end)).select(["time", *target_channels]).sort("time").collect()
    span_desc = "full test split" if args.test_days is None else f"{args.test_days}-day bounded window"
    print(f"Test rows ({span_desc}): {test_df.height}")

    train_values = channel_matrix(train_nominal, target_channels)
    val_values = channel_matrix(val_df, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    spec = WindowSpec(length=10, stride=10)
    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    test_bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    test_labelled_bounds = label_windows(labels_long, test_bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)
    print(f"Test windows: {test_labelled_bounds.height} ({(~test_labelled_bounds['is_nominal']).sum()} non-nominal)")

    val_row_y = row_is_non_nominal(val_df, labels_long, target_channels)
    print(f"Validation rows non-nominal: {int(val_row_y.sum())} / {len(val_row_y)}")

    val_grid_end = val_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    val_bounds = window_bounds_frame(val_df["time"].min(), val_grid_end, config.target_frequency_seconds, spec)
    val_labelled_bounds = label_windows(labels_long, val_bounds, val_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)
    val_window_y = (~val_labelled_bounds.sort("window_index")["is_nominal"]).to_numpy()

    results = {}

    def report(name: str, row_pred, t_elapsed: float, tuned_threshold_info: dict) -> None:
        window_pred = [bool(row_pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
        window_metrics = evaluate_window_predictions(window_pred, test_labelled_bounds, beta=0.5)
        event_summary = summarize_event_detection(evaluate_events(row_pred, test_times, events))
        print(f"\n=== {name} (scored in {t_elapsed:.1f}s) ===")
        print(f"  Tuned on validation: precision={tuned_threshold_info['precision']:.3f} "
              f"recall={tuned_threshold_info['recall']:.3f} f_beta={tuned_threshold_info['f_beta']:.3f}")
        print("  Window-level (test):", window_metrics)
        print("  Event-level detection rate (test):")
        print(event_summary)
        results[name] = window_metrics["f_beta"]

    # 1. WindowedIsolationForest, raw flattened features.
    t0 = time.time()
    iforest_raw = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(train_values)
    tuned = find_best_threshold(iforest_raw.score(val_values), val_window_y, beta=0.5)
    test_window_pred = iforest_raw.score(test_values) > tuned["threshold"]
    test_row_pred = broadcast_window_pred_to_rows(test_window_pred, test_values.shape[0], spec)
    report("WindowedIsolationForest (raw features, tuned threshold)", test_row_pred, time.time() - t0, tuned)

    # 2. WindowedIsolationForest, statistical+spectral features (longer window: spectral
    # features need more than 10 samples to be meaningful).
    spec_feat = WindowSpec(length=50, stride=25)
    t0 = time.time()
    iforest_feat = WindowedIsolationForest(spec_feat, feature_extractor=extract_window_features, n_estimators=100, contamination=0.05).fit(train_values)
    val_bounds_feat = window_bounds_frame(val_df["time"].min(), val_grid_end, config.target_frequency_seconds, spec_feat)
    val_labelled_bounds_feat = label_windows(labels_long, val_bounds_feat, val_df["time"].min(), config.target_frequency_seconds, spec_feat, channels=target_channels)
    val_window_y_feat = (~val_labelled_bounds_feat.sort("window_index")["is_nominal"]).to_numpy()
    tuned_feat = find_best_threshold(iforest_feat.score(val_values), val_window_y_feat, beta=0.5)
    test_window_pred_feat = iforest_feat.score(test_values) > tuned_feat["threshold"]
    test_row_pred_feat = broadcast_window_pred_to_rows(test_window_pred_feat, test_values.shape[0], spec_feat)
    report("WindowedIsolationForest (statistical+spectral features, tuned threshold)", test_row_pred_feat, time.time() - t0, tuned_feat)

    # 3. PCAReconstructor.
    t0 = time.time()
    pca = PCAReconstructor(n_components=0.95).fit(train_values)
    tuned_pca = find_best_threshold(pca.score(val_values), val_row_y, beta=0.5)
    test_pred_pca = pca.score(test_values) > tuned_pca["threshold"]
    report("PCAReconstructor (tuned threshold)", test_pred_pca, time.time() - t0, tuned_pca)

    # 4. Autoencoder (heavy, nonlinear).
    t0 = time.time()
    ae = Autoencoder(n_channels=len(target_channels), hidden_sizes=(32, 16), latent_size=8)
    ae.fit(train_values, epochs=args.ae_epochs, batch_size=2048, verbose=True)
    print(f"Autoencoder trained in {time.time() - t0:.1f}s")
    tuned_ae = find_best_threshold(ae.score(val_values), val_row_y, beta=0.5)
    t0 = time.time()
    test_pred_ae = ae.score(test_values) > tuned_ae["threshold"]
    report("Autoencoder (nonlinear, tuned threshold)", test_pred_ae, time.time() - t0, tuned_ae)

    print("\n=== Summary (test window-level F0.5, best first) ===")
    for name, f in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {f:.4f}  {name}")


if __name__ == "__main__":
    main()
