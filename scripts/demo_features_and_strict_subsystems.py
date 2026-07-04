#!/usr/bin/env python3
"""Two of the "open next steps" from the README, benchmarked against real mission data
without re-running the slower models scripts/run_experiments.py already covers
(PCAReconstructor, raw WindowedIsolationForest, Autoencoder):

1. WindowedIsolationForest with statistical+spectral window features
   (models/window_features.py) instead of raw flattened window values.
2. SubsystemPCAEnsemble with a stricter k-of-n subsystem agreement rule instead of
   plain OR, motivated by the OR rule's confirmed real-data negative result (compounded
   false positives across independent subsystems).

Usage: python scripts/demo_features_and_strict_subsystems.py --mission 2
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
from esa_adb.evaluation import evaluate_event_wise, evaluate_window_predictions, find_best_threshold
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.lightweight import WindowedIsolationForest
from esa_adb.models.subsystem_ensemble import SubsystemPCAEnsemble
from esa_adb.models.window_features import extract_window_features
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--mlflow", action="store_true")
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

    val_df = lf.filter((pl.col("time") >= config.split.val_start) & (pl.col("time") < config.split.train_end)).select(["time", *target_channels]).sort("time").collect()
    val_nominal = filter_nominal_rows(val_df, labels_long, target_channels)

    test_df = lf.filter((pl.col("time") >= config.split.test_start) & (pl.col("time") < config.split.test_end)).select(["time", *target_channels]).sort("time").collect()
    print(f"Test rows (full test split): {test_df.height}")

    train_values = channel_matrix(train_nominal, target_channels)
    val_values = channel_matrix(val_df, target_channels)
    val_nominal_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    spec = WindowSpec(length=10, stride=10)
    val_grid_end = val_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    val_bounds = window_bounds_frame(val_df["time"].min(), val_grid_end, config.target_frequency_seconds, spec)
    val_labelled_bounds = label_windows(labels_long, val_bounds, val_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)
    val_window_y = (~val_labelled_bounds.sort("window_index")["is_nominal"]).to_numpy()

    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    test_bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    test_labelled_bounds = label_windows(labels_long, test_bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)

    def evaluate_and_log(model_name: str, row_pred: np.ndarray, config_dict: dict, t_elapsed: float, notes: str = "") -> None:
        window_pred = [bool(row_pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
        window_metrics = evaluate_window_predictions(window_pred, test_labelled_bounds, beta=0.5)
        event_wise_metrics = evaluate_event_wise(row_pred, test_times, events, beta=0.5)
        run_id = log_experiment(
            model_name=model_name,
            mission=args.mission,
            config=config_dict,
            metrics={"window": window_metrics, "event_wise": event_wise_metrics},
            dataset_scope={
                "train_rows": int(train_values.shape[0]),
                "test_rows": int(test_values.shape[0]),
                "test_start": str(config.split.test_start),
                "test_end": str(config.split.test_end),
            },
            notes=notes,
            mirror_to_mlflow=args.mlflow,
        )
        print(
            f"[{run_id}] {model_name} ({t_elapsed:.1f}s): "
            f"window_f0.5={window_metrics['f_beta']:.3f} event_wise_f0.5={event_wise_metrics['f_beta']:.3f} "
            f"(events {event_wise_metrics['n_events_detected']}/{event_wise_metrics['n_events']}, "
            f"detections {event_wise_metrics['n_detections']})"
        )

    # --- 1: WindowedIsolationForest with statistical+spectral window features ---
    t0 = time.time()
    iforest_feat = WindowedIsolationForest(spec, feature_extractor=extract_window_features, n_estimators=100, contamination=0.05).fit(train_values)
    tuned_feat = find_best_threshold(iforest_feat.score(val_values), val_window_y, beta=0.5)
    iforest_feat_window_pred = iforest_feat.score(test_values) > tuned_feat["threshold"]
    iforest_feat_row_pred = broadcast_window_pred_to_rows(iforest_feat_window_pred, test_values.shape[0], spec)
    evaluate_and_log(
        "WindowedIsolationForest-features", iforest_feat_row_pred,
        {"window_length": spec.length, "window_stride": spec.stride, "contamination": 0.05, "feature_extractor": "extract_window_features"},
        time.time() - t0,
        notes="Open next step: statistical+spectral window features (mean/std/skew/kurtosis/slope/spectral) vs. raw flattened values",
    )

    # --- 2: SubsystemPCAEnsemble, OR baseline + stricter k-of-n agreement ---
    subsystem_col = "subsystem" if "subsystem" in channel_meta.columns else None
    if not subsystem_col:
        print("No subsystem column in channels.csv -- skipping SubsystemPCAEnsemble")
        return

    channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta[subsystem_col]))
    groups: dict[str, list[int]] = {}
    for i, ch in enumerate(target_channels):
        groups.setdefault(channel_to_subsystem.get(ch, "unknown"), []).append(i)

    t0 = time.time()
    subsystem_ensemble = SubsystemPCAEnsemble(groups, min_channels=3, n_components=0.95).fit(train_values)
    subsystem_ensemble.calibrate_threshold(val_nominal_values, percentile=99.0)
    fit_elapsed = time.time() - t0
    n_subsystems = len(subsystem_ensemble.channel_groups)
    print(f"{n_subsystems} subsystems (>=3 channels): {[f'{k}({len(v)})' for k, v in subsystem_ensemble.channel_groups.items()]}")

    t0 = time.time()
    pred_or = subsystem_ensemble.predict(test_values, min_agreeing=1)
    evaluate_and_log(
        "SubsystemPCAEnsemble", pred_or,
        {"n_subsystems": n_subsystems, "min_channels": 3, "n_components": 0.95, "min_agreeing": 1},
        fit_elapsed + (time.time() - t0), notes="Step 2 baseline: one PCA model per channels.csv subsystem, OR-combined (min_agreeing=1)",
    )

    for min_agreeing in sorted({2, max(2, n_subsystems // 2)}):
        if min_agreeing > n_subsystems:
            continue
        t0 = time.time()
        pred_strict = subsystem_ensemble.predict(test_values, min_agreeing=min_agreeing)
        evaluate_and_log(
            f"SubsystemPCAEnsemble-min{min_agreeing}", pred_strict,
            {"n_subsystems": n_subsystems, "min_channels": 3, "n_components": 0.95, "min_agreeing": min_agreeing},
            time.time() - t0,
            notes=f"Open next step: requires {min_agreeing}/{n_subsystems} subsystems to agree instead of plain OR",
        )

    print(f"\nAll runs logged to {Path('data/experiments/results.jsonl').resolve()}")


if __name__ == "__main__":
    main()
