#!/usr/bin/env python3
"""Validates `evaluate_event_wise_corrected` (the ESA-ADB paper's own "corrected

SUPERSEDED by scripts/run_benchmark.py --protocol paper-split (same metric, same split,
shared runner/calibration code). Kept for provenance; its lightweight-subset replication
of the paper's 0.949 remains unique to this script.

event-wise F-score", Kotowski et al. 2024) two ways, both using the paper's own fixed
single train/val/test split (not this project's primary walk-forward methodology --
this script exists specifically to be comparable to the paper's published numbers):

1. **Replication**: `WindowedIsolationForest` on Mission2's exact "lightweight subset"
   (channels 18-28, subsystem 1 -- confirmed directly against the raw channels.csv, not
   assumed) -- the paper's own published corrected event-wise F0.5 there is 0.949. This
   checks whether this project's metric implementation and pipeline land in the same
   ballpark on the same, easier, 11-channel slice the paper used for that number.
2. **Fair full-set comparison**: the same models this project already reports numbers
   for (`PCAReconstructor`, `WindowedIsolationForest`, `Autoencoder`), on the FULL
   47-channel set -- compared against the paper's own full-set best (0.071,
   Telemanom-ESA-Pruned; their own Windowed iForest couldn't even run on the full set,
   confirmed by reading the paper directly), not the frequently-cited but easier-subset
   0.949 number.

Usage: python scripts/demo_corrected_event_wise.py --mission 2
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
from esa_adb.evaluation import evaluate_event_wise, evaluate_event_wise_corrected, find_best_threshold
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.heavy import Autoencoder
from esa_adb.models.lightweight import PCAReconstructor, WindowedIsolationForest
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame

LIGHTWEIGHT_SUBSET_MISSION2 = [f"channel_{i}" for i in range(18, 29)]  # confirmed: subsystem_1, all target


def _load_split(lf, config, target_channels, labels_long):
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
    return train_nominal, val_df, val_nominal, test_df


def _report(model_name: str, mission: int, row_pred: np.ndarray, test_df, target_channels, events, t_elapsed: float, config_dict: dict, notes: str, mlflow: bool):
    test_times = test_df["time"].to_numpy()
    naive = evaluate_event_wise(row_pred, test_times, events, beta=0.5)
    corrected = evaluate_event_wise_corrected(row_pred, test_times, events, beta=0.5)
    run_id = log_experiment(
        model_name=model_name,
        mission=mission,
        config=config_dict,
        metrics={"event_wise": naive, "event_wise_corrected": corrected},
        dataset_scope={"test_rows": int(test_df.height), "n_channels": len(target_channels)},
        notes=notes,
        mirror_to_mlflow=mlflow,
    )
    print(
        f"[{run_id}] {model_name} ({t_elapsed:.1f}s): "
        f"event_wise_f0.5={naive['f_beta']:.3f} corrected_f0.5={corrected['f_beta']:.3f} "
        f"(raw_precision={corrected['raw_precision']:.3f}, tnr_t={corrected['tnr_t']:.4f}, "
        f"events {corrected['n_events_detected']}/{corrected['n_events']}, detections {corrected['n_detections']})"
    )
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[2])  # lightweight subset only confirmed for Mission2
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)
    channel_meta = load_channel_metadata(Path(config.raw_data_dir) / "channels.csv")
    all_target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()

    chunks_glob = str(Path(config.processed_dir) / "chunks" / "*.parquet")
    lf = pl.scan_parquet(chunks_glob)
    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
    events = pl.read_csv(Path(config.interim_dir) / f"anomalies_mission{args.mission}.csv", try_parse_dates=True)

    spec = WindowSpec(length=10, stride=10)

    # --- 1. Replication: WindowedIsolationForest on the lightweight subset (channels 18-28) ---
    print(f"\n=== Replication: WindowedIsolationForest on lightweight subset {LIGHTWEIGHT_SUBSET_MISSION2} ===")
    lw_channels = [c for c in LIGHTWEIGHT_SUBSET_MISSION2 if c in all_target_channels]
    assert lw_channels == LIGHTWEIGHT_SUBSET_MISSION2, f"lightweight subset mismatch: {lw_channels}"
    lw_events = events.filter(pl.col("Channel").is_in(lw_channels))
    train_nominal, val_df, val_nominal, test_df = _load_split(lf, config, lw_channels, labels_long)
    train_values = channel_matrix(train_nominal, lw_channels)
    val_values = channel_matrix(val_df, lw_channels)

    t0 = time.time()
    iforest = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(train_values)
    val_grid_end = val_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    val_bounds = window_bounds_frame(val_df["time"].min(), val_grid_end, config.target_frequency_seconds, spec)
    val_labelled_bounds = label_windows(labels_long, val_bounds, val_df["time"].min(), config.target_frequency_seconds, spec, channels=lw_channels)
    val_window_y = (~val_labelled_bounds.sort("window_index")["is_nominal"]).to_numpy()
    n = min(len(val_window_y), iforest.score(val_values).shape[0])
    tuned = find_best_threshold(iforest.score(val_values)[:n], val_window_y[:n], beta=0.5)
    test_values = channel_matrix(test_df, lw_channels)
    window_pred = iforest.score(test_values) > tuned["threshold"]
    row_pred = broadcast_window_pred_to_rows(window_pred, test_values.shape[0], spec)
    _report(
        "WindowedIsolationForest-lightweight18-28", args.mission, row_pred, test_df, lw_channels, lw_events, time.time() - t0,
        {"window_length": spec.length, "window_stride": spec.stride, "contamination": 0.05, "channels": "18-28"},
        "Replication of paper's Table 2 Mission2 lightweight-subset Windowed iForest result (published: corrected event-wise F0.5=0.949)",
        args.mlflow,
    )

    # --- 2. Fair full-set comparison: same models already reported, full 47-channel set ---
    print(f"\n=== Full-set ({len(all_target_channels)} channels) corrected event-wise, for fair comparison to paper's own full-set best (0.071) ===")
    train_nominal, val_df, val_nominal, test_df = _load_split(lf, config, all_target_channels, labels_long)
    full_events = events.filter(pl.col("Channel").is_in(all_target_channels))
    train_values = channel_matrix(train_nominal, all_target_channels)
    val_values = channel_matrix(val_df, all_target_channels)
    val_nominal_values = channel_matrix(val_nominal, all_target_channels)
    test_values = channel_matrix(test_df, all_target_channels)

    t0 = time.time()
    pca = PCAReconstructor(n_components=0.95).fit(train_values)
    pca.calibrate_threshold(val_nominal_values, percentile=99.0)
    pred_pca = pca.predict(test_values)
    _report(
        "PCAReconstructor", args.mission, pred_pca, test_df, all_target_channels, full_events, time.time() - t0,
        {"n_components": 0.95, "threshold_percentile": 99.0},
        "Fair full-set comparison to paper's own full-set corrected event-wise best (0.071, Telemanom-ESA-Pruned)",
        args.mlflow,
    )

    t0 = time.time()
    iforest_full = WindowedIsolationForest(spec, n_estimators=100, contamination=0.05).fit(train_values)
    val_grid_end = val_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    val_bounds = window_bounds_frame(val_df["time"].min(), val_grid_end, config.target_frequency_seconds, spec)
    val_labelled_bounds = label_windows(labels_long, val_bounds, val_df["time"].min(), config.target_frequency_seconds, spec, channels=all_target_channels)
    val_window_y = (~val_labelled_bounds.sort("window_index")["is_nominal"]).to_numpy()
    n = min(len(val_window_y), iforest_full.score(val_values).shape[0])
    tuned = find_best_threshold(iforest_full.score(val_values)[:n], val_window_y[:n], beta=0.5)
    window_pred = iforest_full.score(test_values) > tuned["threshold"]
    row_pred = broadcast_window_pred_to_rows(window_pred, test_values.shape[0], spec)
    _report(
        "WindowedIsolationForest", args.mission, row_pred, test_df, all_target_channels, full_events, time.time() - t0,
        {"window_length": spec.length, "window_stride": spec.stride, "contamination": 0.05},
        "Fair full-set comparison -- the paper's own Windowed iForest could not run on the full channel set (out-of-memory)",
        args.mlflow,
    )

    t0 = time.time()
    ae = Autoencoder(n_channels=len(all_target_channels), hidden_sizes=(32, 16), latent_size=8)
    ae.fit(train_values, epochs=args.ae_epochs, batch_size=2048, verbose=True)
    ae.calibrate_threshold(val_nominal_values, percentile=99.0)
    pred_ae = ae.predict(test_values)
    _report(
        "Autoencoder", args.mission, pred_ae, test_df, all_target_channels, full_events, time.time() - t0,
        {"hidden_sizes": [32, 16], "latent_size": 8, "epochs": args.ae_epochs},
        "Fair full-set comparison to paper's own full-set corrected event-wise best (0.071, Telemanom-ESA-Pruned)",
        args.mlflow,
    )

    print(f"\nAll runs logged to {Path('data/experiments/results.jsonl').resolve()}")


if __name__ == "__main__":
    main()
