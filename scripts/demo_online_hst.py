#!/usr/bin/env python3
"""Benchmark HalfSpaceTrees (online/incremental isolation-forest-style detector) on real

SUPERSEDED by scripts/run_benchmark.py (HalfSpaceTrees is a first-class model there).
Kept for provenance of the older expanding-history comparison.

mission data, in two ways:

1. Single-split: same test-period benchmark as scripts/run_experiments.py's other
   models, for a directly comparable window/event-wise F0.5 row in the README table.
2. Walk-forward online-vs-refit: the actual point of an incremental model. Two
   HalfSpaceTrees instances walk the same monthly folds --
     - "refit": a brand new HalfSpaceTrees().fit() every fold (structure AND mass reset)
     - "online": one instance built once, updated every fold via partial_fit() only
   -- compared on both fold-time (does skipping the refit actually save wall-clock time?)
   and macro F0.5 (does the mass profile staying in sync with drift lose accuracy vs a
   clean refit?).

Usage: python scripts/demo_online_hst.py --mission 2 [--max-folds N]
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
from esa_adb.models.base import channel_matrix
from esa_adb.models.lightweight.half_space_trees import HalfSpaceTrees
from esa_adb.models.walk_forward import aggregate_fold_results, generate_walk_forward_folds
from esa_adb.preprocessing.labeling import filter_nominal_rows, row_is_non_nominal
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame, window_row_slices


def _load_slice(lf: pl.LazyFrame, start, end, target_channels: list[str]) -> pl.DataFrame:
    return lf.filter((pl.col("time") >= start) & (pl.col("time") < end)).select(["time", *target_channels]).sort("time").collect()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta**2
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f_beta": f_beta, "beta": beta}


def _single_split_benchmark(config, target_channels, lf, labels_long, events, mission: int, use_mlflow: bool) -> None:
    print("\n=== Single-split benchmark (same test period as run_experiments.py) ===")
    train_raw = lf.filter(pl.col("split") == "train").select(["time", *target_channels]).collect()
    train_nominal = filter_nominal_rows(train_raw, labels_long, target_channels)

    val_df = _load_slice(lf, config.split.val_start, config.split.train_end, target_channels)
    val_nominal = filter_nominal_rows(val_df, labels_long, target_channels)

    test_df = _load_slice(lf, config.split.test_start, config.split.test_end, target_channels)

    train_values = channel_matrix(train_nominal, target_channels)
    val_nominal_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    # window_size deliberately larger than train_values -- partial_fit() (called inside
    # fit()) then never auto-swaps, so flush_window() below folds the ENTIRE training
    # split into one reference window instead of just its last few thousand rows (see
    # flush_window()'s docstring).
    t0 = time.time()
    model = HalfSpaceTrees(n_channels=len(target_channels), n_trees=25, max_depth=12, window_size=10_000_000, seed=42)
    model.fit(train_values)
    model.flush_window()
    model.calibrate_threshold(val_nominal_values, percentile=99.0)
    fit_time = time.time() - t0
    pred = model.predict(test_values)

    spec = WindowSpec(length=10, stride=10)
    window_pred = [bool(pred[start:end].any()) for _, start, end in window_row_slices(test_values.shape[0], spec)]
    test_grid_end = test_df["time"].max() + timedelta(seconds=config.target_frequency_seconds)
    test_bounds = window_bounds_frame(test_df["time"].min(), test_grid_end, config.target_frequency_seconds, spec)
    test_labelled_bounds = label_windows(labels_long, test_bounds, test_df["time"].min(), config.target_frequency_seconds, spec, channels=target_channels)
    window_metrics = evaluate_window_predictions(window_pred, test_labelled_bounds, beta=0.5)
    event_wise_metrics = evaluate_event_wise(pred, test_times, events, beta=0.5)

    run_id = log_experiment(
        model_name="HalfSpaceTrees",
        mission=mission,
        config={"n_trees": 25, "max_depth": 12, "window_size": "full_batch_via_flush_window", "threshold_percentile": 99.0},
        metrics={"window": window_metrics, "event_wise": event_wise_metrics},
        dataset_scope={
            "train_rows": int(train_values.shape[0]),
            "test_rows": int(test_values.shape[0]),
            "test_start": str(config.split.test_start),
            "test_end": str(config.split.test_end),
        },
        notes="Open next step 1: online/incremental Isolation-Forest-style detector, fit()+calibrate_threshold() single-split benchmark",
        mirror_to_mlflow=use_mlflow,
    )
    print(
        f"[{run_id}] HalfSpaceTrees ({fit_time:.1f}s fit): "
        f"window_f0.5={window_metrics['f_beta']:.3f} event_wise_f0.5={event_wise_metrics['f_beta']:.3f} "
        f"(events {event_wise_metrics['n_events_detected']}/{event_wise_metrics['n_events']}, "
        f"detections {event_wise_metrics['n_detections']})"
    )


def _calibrate(model: HalfSpaceTrees, calib_df: pl.DataFrame, target_channels: list[str], labels_long: pl.DataFrame) -> None:
    """Threshold from the fold's OWN held-out calibration sub-period (never the training
    rows that just fed the mass profile) -- the same fit/calibration split
    scripts/demo_walk_forward.py's `_fit_pca` uses, and for the identical reason: a
    percentile of the model's own training scores badly underestimates real error/
    novelty and can produce a threshold too high for any row to ever exceed.
    """
    calib_values = channel_matrix(calib_df, target_channels)
    calib_y = row_is_non_nominal(calib_df, labels_long, target_channels)
    if calib_y.sum() > 0:
        tuned = find_best_threshold(model.score(calib_values), calib_y, beta=0.5)
        model.threshold_ = tuned["threshold"] if tuned["f_beta"] > 0 else float(np.percentile(model.score(calib_values), 99.0))
    else:
        model.threshold_ = float(np.percentile(model.score(calib_values), 99.0))


def _walk_forward_online_vs_refit(config, target_channels, lf, labels_long, mission: int, max_folds, use_mlflow: bool) -> None:
    print("\n=== Walk-forward: online partial_fit vs. full refit per fold ===")
    folds = list(generate_walk_forward_folds(config.split.test_start, config.split.test_end, step_months=1, window="expanding"))
    if max_folds:
        folds = folds[:max_folds]
    print(f"{len(folds)} monthly folds over the test period")

    n_channels = len(target_channels)
    online_model = None
    refit_results, online_results = [], []
    refit_time_total, online_time_total = 0.0, 0.0

    for fold in folds:
        train_df = _load_slice(lf, fold.train_start, fold.train_end, target_channels)
        eval_df = _load_slice(lf, fold.eval_start, fold.eval_end, target_channels)
        if train_df.height < 300 or eval_df.height < 200:
            continue

        # Split this fold's OWN training period into a fit sub-period (first 2/3) and a
        # calibration sub-period (last 1/3) -- same split _fit_pca uses in
        # demo_walk_forward.py, for the same reason (see _calibrate's docstring).
        split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
        fit_df = train_df.filter(pl.col("time") < split_point)
        calib_df = train_df.filter(pl.col("time") >= split_point)
        fit_nominal = filter_nominal_rows(fit_df, labels_long, target_channels)
        if fit_nominal.height < 300 or calib_df.height < 200:
            continue
        fit_values = channel_matrix(fit_nominal, target_channels)

        eval_values = channel_matrix(eval_df, target_channels)
        eval_y = row_is_non_nominal(eval_df, labels_long, target_channels)

        # "refit": brand new model every fold, trained on all history seen so far (mirrors
        # the "expanding" walk-forward variant already used for PCAReconstructor).
        # window_size larger than any single fold means partial_fit() (called inside
        # fit()) never auto-swaps -- flush_window() folds the WHOLE fit sub-period into
        # one reference window (see HalfSpaceTrees.flush_window's docstring).
        t0 = time.time()
        refit_model = HalfSpaceTrees(n_channels=n_channels, n_trees=25, max_depth=12, window_size=10_000_000, seed=42)
        refit_model.fit(fit_values)
        refit_model.flush_window()
        _calibrate(refit_model, calib_df, target_channels, labels_long)
        refit_time_total += time.time() - t0
        refit_pred = refit_model.predict(eval_values)
        refit_results.append(_metrics(eval_y, refit_pred))

        # "online": only the FIRST fold pays a fit() cost; every later fold streams its
        # own new nominal rows into the SAME instance via partial_fit -- no refit at all.
        # Each fold's flush_window() replaces ref_mass with just THAT fold's fit
        # sub-period (a "reactive," bounded-recency profile, same spirit as the
        # PCAReconstructor "sliding" walk-forward variant) rather than accumulating
        # unboundedly -- partial_fit only ever needs to reflect the current regime.
        t0 = time.time()
        if online_model is None:
            online_model = HalfSpaceTrees(n_channels=n_channels, n_trees=25, max_depth=12, window_size=10_000_000, seed=42)
            online_model.fit(fit_values)
        else:
            online_model.partial_fit(fit_values)
        online_model.flush_window()
        _calibrate(online_model, calib_df, target_channels, labels_long)
        online_time_total += time.time() - t0
        online_pred = online_model.predict(eval_values)
        online_results.append(_metrics(eval_y, online_pred))

        print(
            f"fold {fold.fold_index}: {fold.eval_start.date()}..{fold.eval_end.date()} "
            f"refit_f0.5={refit_results[-1]['f_beta']:.3f} online_f0.5={online_results[-1]['f_beta']:.3f}"
        )

    for variant, results, elapsed in [("refit", refit_results, refit_time_total), ("online", online_results, online_time_total)]:
        agg = aggregate_fold_results(results)
        print(f"\n=== HalfSpaceTrees-{variant}: {agg['n_folds']} folds, {elapsed:.1f}s total fit/update time ===")
        print("  micro:", agg["micro"])
        print("  macro F0.5:", agg["macro"].get("f_beta"))
        run_id = log_experiment(
            model_name=f"HalfSpaceTrees-{variant}",
            mission=mission,
            config={"n_trees": 25, "max_depth": 12, "window_size": "one_fold_via_flush_window", "step_months": 1},
            metrics={"micro": agg["micro"], "macro": agg["macro"], "fit_time_seconds": elapsed},
            dataset_scope={"n_folds": agg["n_folds"], "test_start": str(config.split.test_start), "test_end": str(config.split.test_end)},
            notes=(
                "Open next step 1: online/incremental detector -- "
                + ("brand-new fit() every fold (baseline)" if variant == "refit" else "partial_fit() only after the first fold, no refit")
            ),
            mirror_to_mlflow=use_mlflow,
        )
        print(f"  logged as [{run_id}]")

    speedup = refit_time_total / online_time_total if online_time_total > 0 else float("nan")
    print(f"\nOnline update was {speedup:.1f}x faster than a full refit per fold ({refit_time_total:.1f}s vs {online_time_total:.1f}s total)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--max-folds", type=int, default=None, help="default: no cap, covers the full test period")
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

    _single_split_benchmark(config, target_channels, lf, labels_long, events, args.mission, args.mlflow)
    _walk_forward_online_vs_refit(config, target_channels, lf, labels_long, args.mission, args.max_folds, args.mlflow)

    print(f"\nAll runs logged to {Path('data/experiments/results.jsonl').resolve()}")


if __name__ == "__main__":
    main()
