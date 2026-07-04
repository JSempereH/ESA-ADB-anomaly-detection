#!/usr/bin/env python3
"""Run a batch of experiments against a mission's real processed data, logging every
run via esa_adb.experiments.tracking.log_experiment to data/experiments/results.jsonl.

Usage: python scripts/run_experiments.py --mission 2 [--ae-epochs 30] [--ae-train-days N]

Covers three of the four "next steps" from the walk-forward writeup (the fourth, longer
sliding-window walk-forward, is scripts/demo_walk_forward.py --sliding-months 3 --
also wired to log here via the same tracker):
  - re-evaluates the existing best models with the event-wise metric (closer to the
    paper's corrected event-wise scoring) alongside the existing window-level one
  - SubsystemPCAEnsemble: one PCA model per channels.csv subsystem group instead of one
    joint model across all 47 target channels
  - Autoencoder trained with more epochs on the FULL train split (vs. the earlier
    10-day/5-epoch CPU smoke-test) -- still bounded by this machine's CPU, not the GPU
    the project's own docs say real training benefits from, but pushed further within
    that constraint
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
from esa_adb.evaluation import (
    evaluate_event_wise,
    evaluate_window_predictions,
    find_best_threshold,
)
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.heavy import Autoencoder
from esa_adb.models.lightweight import PCAReconstructor, WindowedIsolationForest
from esa_adb.models.subsystem_ensemble import SubsystemPCAEnsemble
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--ae-train-days", type=int, default=None, help="bound AE train data (default: full train split)")
    parser.add_argument("--mlflow", action="store_true", help="also mirror every logged run into MLflow (sqlite:///mlflow.db)")
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
    val_df = lf.filter((pl.col("time") >= val_start) & (pl.col("time") < config.split.train_end)).select(["time", *target_channels]).sort("time").collect()
    val_nominal = filter_nominal_rows(val_df, labels_long, target_channels)
    print(f"Validation rows: {val_df.height} ({val_nominal.height} nominal)")

    test_df = lf.filter((pl.col("time") >= config.split.test_start) & (pl.col("time") < config.split.test_end)).select(["time", *target_channels]).sort("time").collect()
    print(f"Test rows (full test split): {test_df.height}")

    train_values = channel_matrix(train_nominal, target_channels)
    val_values = channel_matrix(val_df, target_channels)
    val_nominal_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    spec = WindowSpec(length=10, stride=10)

    # WindowedIsolationForest scores are one-per-window, not one-per-row -- needs its
    # own window-level ground truth for find_best_threshold (PCA/Autoencoder/Subsystem
    # models below use calibrate_threshold's percentile-based approach instead, since
    # they score per-row and don't need this).
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

    # --- Step 4: re-evaluate existing best models with the event-wise metric too ---
    t0 = time.time()
    pca = PCAReconstructor(n_components=0.95).fit(train_values)
    pca.calibrate_threshold(val_nominal_values, percentile=99.0)
    pred_pca = pca.predict(test_values)
    evaluate_and_log("PCAReconstructor", pred_pca, {"n_components": 0.95, "threshold_percentile": 99.0}, time.time() - t0,
                      notes="Step 4: re-evaluated with event-wise metric alongside window-level")

    t0 = time.time()
    iforest = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(train_values)
    tuned = find_best_threshold(iforest.score(val_values), val_window_y, beta=0.5)
    iforest_window_pred = iforest.score(test_values) > tuned["threshold"]
    iforest_row_pred = broadcast_window_pred_to_rows(iforest_window_pred, test_values.shape[0], spec)
    evaluate_and_log("WindowedIsolationForest", iforest_row_pred, {"window_length": spec.length, "window_stride": spec.stride, "contamination": 0.05}, time.time() - t0,
                      notes="Step 4: re-evaluated with event-wise metric alongside window-level")

    # --- Step 2: per-subsystem PCA ensemble ---
    # (statistical+spectral window features and stricter k-of-n subsystem agreement --
    # two more "open next steps" -- are benchmarked in
    # scripts/demo_features_and_strict_subsystems.py instead of here, since they don't
    # need this script's other, much slower models re-trained alongside them)
    subsystem_col = "subsystem" if "subsystem" in channel_meta.columns else None
    if subsystem_col:
        channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta[subsystem_col]))
        groups: dict[str, list[int]] = {}
        for i, ch in enumerate(target_channels):
            groups.setdefault(channel_to_subsystem.get(ch, "unknown"), []).append(i)
        t0 = time.time()
        subsystem_ensemble = SubsystemPCAEnsemble(groups, min_channels=3, n_components=0.95).fit(train_values)
        subsystem_ensemble.calibrate_threshold(val_nominal_values, percentile=99.0)
        pred_subsystem = subsystem_ensemble.predict(test_values)
        evaluate_and_log(
            "SubsystemPCAEnsemble", pred_subsystem,
            {"n_subsystems": len(subsystem_ensemble.channel_groups), "min_channels": 3, "n_components": 0.95,
             "subsystem_sizes": {k: len(v) for k, v in subsystem_ensemble.channel_groups.items()}},
            time.time() - t0, notes="Step 2: one PCA model per channels.csv subsystem, OR-combined",
        )
    else:
        print("No subsystem column in channels.csv -- skipping SubsystemPCAEnsemble")

    # --- Step 3: Autoencoder with more compute (more epochs, more/full train data) ---
    if args.ae_train_days is not None:
        ae_train_end = min(config.split.val_start, config.split.train_start + timedelta(days=args.ae_train_days))
        ae_train_df = lf.filter((pl.col("time") >= config.split.train_start) & (pl.col("time") < ae_train_end)).select(["time", *target_channels]).sort("time").collect()
        ae_train_nominal = filter_nominal_rows(ae_train_df, labels_long, target_channels)
        ae_train_values = channel_matrix(ae_train_nominal, target_channels)
    else:
        ae_train_values = train_values

    t0 = time.time()
    ae = Autoencoder(n_channels=len(target_channels), hidden_sizes=(32, 16), latent_size=8)
    ae.fit(ae_train_values, epochs=args.ae_epochs, batch_size=2048, verbose=True)
    ae.calibrate_threshold(val_nominal_values, percentile=99.0)
    fit_time = time.time() - t0
    print(f"Autoencoder trained on {ae_train_values.shape[0]} rows for {args.ae_epochs} epochs in {fit_time:.1f}s")
    pred_ae = ae.predict(test_values)
    evaluate_and_log(
        "Autoencoder", pred_ae,
        {"hidden_sizes": [32, 16], "latent_size": 8, "epochs": args.ae_epochs, "train_rows": int(ae_train_values.shape[0])},
        fit_time, notes="Step 3: more compute than the original 10-day/5-epoch CPU smoke-test",
    )

    print(f"\nAll runs logged to {Path('data/experiments/results.jsonl').resolve()}")


if __name__ == "__main__":
    main()
