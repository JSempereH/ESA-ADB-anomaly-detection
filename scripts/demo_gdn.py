#!/usr/bin/env python3
"""Single-split benchmark for `GDN` (Graph Deviation Network), matching
scripts/run_experiments.py's pattern for the other models: train on the full nominal
train split, calibrate on held-out nominal validation data, evaluate on the full test
split with window-level, event-wise, and corrected event-wise metrics. Also prints the
learned graph's edges for a handful of channels, as a first look at GDN's explainability
story on real data.

Usage: python scripts/demo_gdn.py --mission 2 [--epochs 15]
"""
import argparse
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.evaluation import evaluate_event_wise, evaluate_event_wise_corrected, evaluate_window_predictions
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import channel_matrix
from esa_adb.models.heavy import GDN
from esa_adb.preprocessing.labeling import filter_nominal_rows
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="training window stride -- GDN.fit()'s default stride=1 builds one window per row (fully "
        "overlapping); on Mission2's 2.4M-row train split that's a ~9GB array for X alone, easily enough "
        "to OOM a laptop, the same stride=1 memory blowup the paper's own Windowed iForest hits (see "
        "README). 10 matches this project's WindowSpec convention elsewhere (~0.9GB).",
    )
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
    val_nominal_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    t0 = time.time()
    model = GDN(n_channels=n_channels, window_length=10, embed_dim=32, top_k=15, hidden_size=64)
    model.fit(train_values, epochs=args.epochs, batch_size=2048, stride=args.stride, verbose=True)
    model.calibrate_threshold(val_nominal_values, percentile=99.0)
    fit_time = time.time() - t0
    print(f"GDN trained on {train_values.shape[0]} rows for {args.epochs} epochs in {fit_time:.1f}s")

    print("\nLearned graph, first 10 target channels' neighbors:")
    edges = model.graph_edges(channel_names=target_channels)
    edges_by_source: dict[str, list[str]] = {}
    for src, dst in edges:
        edges_by_source.setdefault(src, []).append(dst)
    for ch in target_channels[:10]:
        print(f"  {ch} -> {edges_by_source.get(ch, [])}")

    graph_vs_subsystem = None
    if "subsystem" in channel_meta.columns:
        channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta["subsystem"]))
        same_subsystem = sum(1 for src, dst in edges if channel_to_subsystem.get(src) == channel_to_subsystem.get(dst))
        observed_fraction = same_subsystem / len(edges)
        subsystem_sizes = Counter(channel_to_subsystem.get(c) for c in target_channels)
        expected_fraction = float(
            np.mean([(subsystem_sizes[channel_to_subsystem[c]] - 1) / (n_channels - 1) for c in target_channels])
        )
        graph_vs_subsystem = {
            "observed_same_subsystem_fraction": observed_fraction,
            "expected_same_subsystem_fraction": expected_fraction,
            "ratio": observed_fraction / expected_fraction,
        }
        print(
            f"\nGraph vs. subsystem structure: observed={observed_fraction:.3f}, "
            f"expected(random)={expected_fraction:.3f}, ratio={graph_vs_subsystem['ratio']:.2f}x"
        )

    pred = model.predict(test_values)

    spec = WindowSpec(length=10, stride=10)
    window_pred = [bool(pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    test_bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    test_labelled_bounds = label_windows(labels_long, test_bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)
    window_metrics = evaluate_window_predictions(window_pred, test_labelled_bounds, beta=0.5)
    event_wise_metrics = evaluate_event_wise(pred, test_times, events, beta=0.5)
    corrected_metrics = evaluate_event_wise_corrected(pred, test_times, events, beta=0.5)

    run_id = log_experiment(
        model_name="GDN",
        mission=args.mission,
        config={"window_length": 10, "embed_dim": 32, "top_k": 15, "hidden_size": 64, "epochs": args.epochs, "threshold_percentile": 99.0},
        metrics={
            "window": window_metrics,
            "event_wise": event_wise_metrics,
            "event_wise_corrected": corrected_metrics,
            "graph_vs_subsystem": graph_vs_subsystem,
        },
        dataset_scope={
            "train_rows": int(train_values.shape[0]),
            "test_rows": int(test_values.shape[0]),
            "test_start": str(config.split.test_start),
            "test_end": str(config.split.test_end),
        },
        notes="GDN (Graph Deviation Network, Deng & Hooi AAAI 2021) single-split benchmark, full channel set",
        mirror_to_mlflow=args.mlflow,
    )
    print(
        f"\n[{run_id}] GDN ({fit_time:.1f}s fit): "
        f"window_f0.5={window_metrics['f_beta']:.3f} event_wise_f0.5={event_wise_metrics['f_beta']:.3f} "
        f"corrected_f0.5={corrected_metrics['f_beta']:.3f} "
        f"(events {event_wise_metrics['n_events_detected']}/{event_wise_metrics['n_events']}, "
        f"detections {event_wise_metrics['n_detections']})"
    )


if __name__ == "__main__":
    main()
