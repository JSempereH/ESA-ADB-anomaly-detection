#!/usr/bin/env python3
"""Compares three threshold STRATEGIES on top of the already-established best training
methodology ("online", never-forgetting `partial_fit` -- see
scripts/demo_full_mission_walk_forward.py) for every model in the single-split lineup,
over the FULL Mission2 test period:

  - "static": one threshold, calibrated once per fold from that fold's calibration
    sub-period, applied unchanged across the whole next month (what every walk-forward
    script in this project has done so far).
  - "rolling": `RollingPercentileThreshold` -- a bounded window of recent scores,
    re-percentiled continuously as the eval month streams in, instead of frozen at the
    start of it.
  - "pot": `POTThreshold` -- streaming Peaks-Over-Threshold/Extreme Value Theory (Siffer
    et al., KDD 2017), continuously refitting a Generalized Pareto tail instead of
    tracking a fixed percentile.

Each fold's model is trained/updated via `ModelRunner.online_raw()` -- imported from
demo_full_mission_walk_forward.py -- exactly ONCE per fold, then all three threshold
strategies are applied to that SAME model's calibration/eval scores. Training cost is
paid once; only the (cheap) thresholding differs.

Usage: python scripts/demo_dynamic_threshold.py --mission 2 [--max-folds N] [--models ...]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.dynamic_threshold import POTThreshold, RollingPercentileThreshold
from esa_adb.models.walk_forward import aggregate_fold_results, generate_walk_forward_folds
from esa_adb.preprocessing.labeling import filter_nominal_rows, row_is_non_nominal
from esa_adb.preprocessing.windowing import WindowSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_full_mission_walk_forward import (  # noqa: E402
    ALL_MODELS,
    AutoencoderRunner,
    GlobalSTDRunner,
    LSTMRunner,
    PCARunner,
    SubsystemRunner,
    WindowedIForestRunner,
    _load_slice,
    _metrics,
    _threshold,
)

THRESHOLD_MODES = ["static", "rolling", "pot"]


def _apply_threshold_mode(mode: str, calib_scores: np.ndarray, calib_y, eval_scores: np.ndarray) -> np.ndarray:
    if mode == "static":
        thr = _threshold(calib_scores, calib_y) if calib_y is not None else float(np.percentile(calib_scores, 99.0))
        return eval_scores > thr
    if mode == "rolling":
        rt = RollingPercentileThreshold(window_size=5000, percentile=99.0, initial_scores=calib_scores, recompute_every=200)
        return rt.predict_batch(eval_scores)
    if mode == "pot":
        # recompute_every much larger than rolling's -- genpareto.fit's MLE optimization
        # is expensive enough (see POTThreshold's docstring) that a smaller value would
        # add up fast across many folds/models in a walk-forward comparison; 2000 keeps
        # a still-reasonable cadence of adaptivity at a tractable total cost.
        pt = POTThreshold(risk=1e-3, initial_scores=calib_scores, recompute_every=2000)
        return pt.predict_batch(eval_scores)
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--max-folds", type=int, default=None, help="default: no cap, covers the full test period")
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS))
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    requested = args.models.split(",")

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

    spec = WindowSpec(length=10, stride=10)
    subsystem_col = "subsystem" if "subsystem" in channel_meta.columns else None
    channel_groups = {}
    if subsystem_col:
        channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta[subsystem_col]))
        for i, ch in enumerate(target_channels):
            channel_groups.setdefault(channel_to_subsystem.get(ch, "unknown"), []).append(i)

    runners = {}
    if "GlobalSTD" in requested:
        runners["GlobalSTD"] = GlobalSTDRunner(target_channels, labels_long, config)
    if "PCAReconstructor" in requested:
        runners["PCAReconstructor"] = PCARunner(target_channels, labels_long, config)
    if "WindowedIsolationForest" in requested:
        runners["WindowedIsolationForest"] = WindowedIForestRunner(target_channels, labels_long, config, spec)
    if "SubsystemPCAEnsemble" in requested and channel_groups:
        runners["SubsystemPCAEnsemble"] = SubsystemRunner(target_channels, labels_long, config, channel_groups)
    if "Autoencoder" in requested:
        runners["Autoencoder"] = AutoencoderRunner(target_channels, labels_long, config, n_channels)
    if "LSTMForecaster" in requested:
        runners["LSTMForecaster"] = LSTMRunner(target_channels, labels_long, config, n_channels)
    print(f"Models: {list(runners)}")

    folds = list(
        generate_walk_forward_folds(config.split.test_start, config.split.test_end, step_months=1, window="sliding", sliding_window_months=1)
    )
    if args.max_folds:
        folds = folds[: args.max_folds]
    print(f"{len(folds)} monthly folds over the FULL test period ({config.split.test_start.date()} .. {config.split.test_end.date()})")

    results: dict[str, dict[str, list[dict]]] = {name: {mode: [] for mode in THRESHOLD_MODES} for name in runners}
    t_start = time.time()

    for fold in folds:
        train_df = _load_slice(lf, fold.train_start, fold.train_end, target_channels)
        eval_df = _load_slice(lf, fold.eval_start, fold.eval_end, target_channels)
        if train_df.height < 300 or eval_df.height < 200:
            continue

        split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
        fit_df = train_df.filter(pl.col("time") < split_point)
        calib_df = train_df.filter(pl.col("time") >= split_point)
        fit_nominal = filter_nominal_rows(fit_df, labels_long, target_channels)
        if fit_nominal.height < 300 or calib_df.height < 200:
            continue
        fit_values = channel_matrix(fit_nominal, target_channels)
        eval_values = channel_matrix(eval_df, target_channels)
        eval_y = row_is_non_nominal(eval_df, labels_long, target_channels)

        fold_t0 = time.time()
        for name, runner in runners.items():
            t0 = time.time()
            calib_scores, calib_y, eval_scores, is_window = runner.online_raw(fit_values, calib_df, eval_values)
            t_train = time.time() - t0

            line = f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) {name} ({t_train:.1f}s train):"
            for mode in THRESHOLD_MODES:
                pred = _apply_threshold_mode(mode, calib_scores, calib_y, eval_scores)
                if is_window:
                    pred = broadcast_window_pred_to_rows(pred, eval_values.shape[0], spec)
                m = _metrics(eval_y, pred)
                results[name][mode].append(m)
                line += f" {mode}_f0.5={m['f_beta']:.3f}"
            print(line)
        print(f"  -- fold {fold.fold_index} all models: {time.time() - fold_t0:.1f}s")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    for name in runners:
        for mode in THRESHOLD_MODES:
            agg = aggregate_fold_results(results[name][mode])
            print(f"\n=== {name}-online-{mode}: {agg['n_folds']} folds ===")
            print("  micro:", agg["micro"])
            print("  macro F0.5:", agg["macro"].get("f_beta"))
            run_id = log_experiment(
                model_name=f"{name}-online-{mode}",
                mission=args.mission,
                config={"step_months": 1, "fold_definition": "sliding_1_month", "threshold_mode": mode},
                metrics={"micro": agg["micro"], "macro": agg["macro"]},
                dataset_scope={"n_folds": agg["n_folds"], "test_start": str(config.split.test_start), "test_end": str(config.split.test_end)},
                notes=f"Dynamic threshold comparison: {mode} threshold strategy on top of 'online' (never-forgetting) training for {name}",
                mirror_to_mlflow=args.mlflow,
            )
            print(f"  logged as [{run_id}]")


if __name__ == "__main__":
    main()
