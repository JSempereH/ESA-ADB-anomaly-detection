#!/usr/bin/env python3
"""Every lightweight model, on the paper's own fixed single train/val/test split (not

SUPERSEDED by scripts/run_benchmark.py --protocol paper-split. Kept for provenance of
the older logged results.

this project's primary walk-forward methodology), with the SAME three metrics for all of
them: window-level, event-wise, and the paper's own corrected event-wise F-score.

The existing single-split numbers in the README were computed by several different
scripts over time, each reporting whichever metrics it happened to compute -- several
models were missing corrected event-wise entirely. This script re-runs all of them
through one consistent evaluation path so they're directly comparable to each other AND
to the paper's own published numbers, matching scripts/demo_corrected_event_wise.py's
approach but for the full lightweight lineup in one place.

Deliberately NOT the walk-forward/online numbers already reported elsewhere in the
README -- this is the paper-comparable baseline specifically.

Usage: python scripts/demo_lightweight_paper_split.py --mission 2
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
    evaluate_event_wise_corrected,
    evaluate_window_predictions,
    find_best_threshold,
)
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.lightweight import GlobalSTD, HalfSpaceTrees, PCAReconstructor, WindowedIsolationForest
from esa_adb.models.subsystem_ensemble import SubsystemPCAEnsemble
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
    n_channels = len(target_channels)
    print(f"Mission{args.mission}: {n_channels} target channels")

    chunks_glob = str(Path(config.processed_dir) / "chunks" / "*.parquet")
    lf = pl.scan_parquet(chunks_glob)
    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
    events = pl.read_csv(Path(config.interim_dir) / f"anomalies_mission{args.mission}.csv", try_parse_dates=True).filter(
        pl.col("Channel").is_in(target_channels)
    )

    train_raw = lf.filter(pl.col("split") == "train").select(["time", *target_channels]).collect()
    train_nominal = filter_nominal_rows(train_raw, labels_long, target_channels)
    print(f"Train nominal rows: {train_nominal.height} / {train_raw.height}")

    val_df = (
        lf.filter((pl.col("time") >= config.split.val_start) & (pl.col("time") < config.split.train_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    val_nominal = filter_nominal_rows(val_df, labels_long, target_channels)
    print(f"Validation rows: {val_df.height} ({val_nominal.height} nominal)")

    test_df = (
        lf.filter((pl.col("time") >= config.split.test_start) & (pl.col("time") < config.split.test_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
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

    results: list[tuple[str, dict]] = []

    def report(model_name: str, row_pred: np.ndarray, config_dict: dict, t_elapsed: float, notes: str) -> None:
        window_pred = [bool(row_pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
        window_metrics = evaluate_window_predictions(window_pred, test_labelled_bounds, beta=0.5)
        event_wise_metrics = evaluate_event_wise(row_pred, test_times, events, beta=0.5)
        corrected_metrics = evaluate_event_wise_corrected(row_pred, test_times, events, beta=0.5)
        run_id = log_experiment(
            model_name=model_name,
            mission=args.mission,
            config=config_dict,
            metrics={"window": window_metrics, "event_wise": event_wise_metrics, "event_wise_corrected": corrected_metrics},
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
            f"corrected_f0.5={corrected_metrics['f_beta']:.3f}"
        )
        results.append((model_name, {"window": window_metrics["f_beta"], "event_wise": event_wise_metrics["f_beta"], "corrected": corrected_metrics["f_beta"]}))

    notes = "Paper-comparable single-split lightweight-model sweep, consistent window/event-wise/corrected event-wise metrics across all models"

    # --- GlobalSTD ---
    t0 = time.time()
    global_std = GlobalSTD(n_std=3.0).fit(train_values)
    pred = global_std.predict(test_values)
    report("GlobalSTD", pred, {"n_std": 3.0}, time.time() - t0, notes)

    # --- HalfSpaceTrees ---
    t0 = time.time()
    hst = HalfSpaceTrees(n_channels=n_channels, n_trees=25, max_depth=12, window_size=10_000_000, seed=42)
    hst.fit(train_values)
    hst.flush_window()
    hst.calibrate_threshold(val_nominal_values, percentile=99.0)
    pred = hst.predict(test_values)
    report("HalfSpaceTrees", pred, {"n_trees": 25, "max_depth": 12, "window_size": "full_batch_via_flush_window", "threshold_percentile": 99.0}, time.time() - t0, notes)

    # --- WindowedIsolationForest ---
    t0 = time.time()
    iforest = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(train_values)
    tuned = find_best_threshold(iforest.score(val_values), val_window_y, beta=0.5)
    window_pred = iforest.score(test_values) > tuned["threshold"]
    pred = broadcast_window_pred_to_rows(window_pred, test_values.shape[0], spec)
    report("WindowedIsolationForest", pred, {"window_length": spec.length, "window_stride": spec.stride, "contamination": 0.05}, time.time() - t0, notes)

    # --- SubsystemPCAEnsemble (OR and stricter k-of-n) ---
    subsystem_col = "subsystem" if "subsystem" in channel_meta.columns else None
    if subsystem_col:
        channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta[subsystem_col]))
        groups: dict[str, list[int]] = {}
        for i, ch in enumerate(target_channels):
            groups.setdefault(channel_to_subsystem.get(ch, "unknown"), []).append(i)
        t0 = time.time()
        subsystem_ensemble = SubsystemPCAEnsemble(groups, min_channels=3, n_components=0.95).fit(train_values)
        subsystem_ensemble.calibrate_threshold(val_nominal_values, percentile=99.0)
        fit_elapsed = time.time() - t0
        n_subsystems = len(subsystem_ensemble.channel_groups)

        t0 = time.time()
        pred = subsystem_ensemble.predict(test_values, min_agreeing=1)
        report(
            "SubsystemPCAEnsemble-OR", pred,
            {"n_subsystems": n_subsystems, "min_channels": 3, "n_components": 0.95, "min_agreeing": 1},
            fit_elapsed + (time.time() - t0), notes,
        )

        min_agreeing = max(2, n_subsystems // 2)
        t0 = time.time()
        pred = subsystem_ensemble.predict(test_values, min_agreeing=min_agreeing)
        report(
            f"SubsystemPCAEnsemble-min{min_agreeing}", pred,
            {"n_subsystems": n_subsystems, "min_channels": 3, "n_components": 0.95, "min_agreeing": min_agreeing},
            fit_elapsed + (time.time() - t0), notes,
        )
    else:
        print("No subsystem column in channels.csv -- skipping SubsystemPCAEnsemble")

    # --- PCAReconstructor ---
    t0 = time.time()
    pca = PCAReconstructor(n_components=0.95).fit(train_values)
    pca.calibrate_threshold(val_nominal_values, percentile=99.0)
    pred = pca.predict(test_values)
    report("PCAReconstructor", pred, {"n_components": 0.95, "threshold_percentile": 99.0}, time.time() - t0, notes)

    print("\n=== Summary (paper-comparable single split, all consistent metrics) ===")
    print(f"{'Model':<32} {'window':>8} {'event_wise':>12} {'corrected':>10}")
    for name, m in results:
        print(f"{name:<32} {m['window']:>8.3f} {m['event_wise']:>12.3f} {m['corrected']:>10.3f}")

    print(f"\nAll runs logged to {Path('data/experiments/results.jsonl').resolve()}")


if __name__ == "__main__":
    main()
