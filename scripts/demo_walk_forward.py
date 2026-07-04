#!/usr/bin/env python3
"""Walk-forward evaluation: train on one month, evaluate on the next, slide forward.

Usage: python scripts/demo_walk_forward.py --mission 2 [--step-months 1] [--sliding-months 1] [--max-folds N]

Three variants compared over the same monthly folds spanning the test period:
  - expanding (stable): PCAReconstructor retrained each fold on ALL history seen so far
  - sliding (reactive): PCAReconstructor retrained each fold on only the last
    --sliding-months of history
  - hybrid: PairedLearnerEnsemble(stable=expanding, reactive=sliding), AND-combined --
    flags a point only when both agree, deliberately not alarming when only the stable
    (long-memory) model flags something the reactive (short-memory) one has already
    adapted to (settled drift, not an ongoing anomaly)

Literature: Bach & Maloof, "Paired Learners for Concept Drift" (IEEE, 2008) -- the
stable/reactive pairing the hybrid mirrors, adapted from drift *detection* to anomaly
*scoring* (see models/ensemble.py's docstring for the adaptation).

Threshold calibration splits each fold's OWN training period into a fit sub-period
(first 2/3) and a calibration sub-period (last 1/3, labeled, via
evaluation.find_best_threshold) -- never on the fit sub-period's own residuals, since
training residuals systematically underestimate real error on unseen data (documented on
LSTMForecaster too).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from esa_adb.config import load_mission_config
from esa_adb.evaluation import find_best_threshold
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import channel_matrix
from esa_adb.models.ensemble import PairedLearnerEnsemble
from esa_adb.models.lightweight import PCAReconstructor
from esa_adb.models.walk_forward import aggregate_fold_results, generate_walk_forward_folds
from esa_adb.preprocessing.labeling import filter_nominal_rows, row_is_non_nominal


def _load_slice(lf: pl.LazyFrame, start, end, target_channels: list[str]) -> pl.DataFrame:
    return lf.filter((pl.col("time") >= start) & (pl.col("time") < end)).select(["time", *target_channels]).sort("time").collect()


def _fit_pca(train_df: pl.DataFrame, target_channels: list[str], labels_long: pl.DataFrame):
    """Split the fold's own training period into a fit sub-period (first 2/3) and a
    calibration sub-period (last 1/3), both strictly before the eval period. Threshold
    is tuned on the calibration sub-period's *raw* (labeled, not nominal-filtered) rows
    via `find_best_threshold` -- never on the fit sub-period's own training residuals,
    which systematically underestimate real error on unseen data (the same trap
    documented on `LSTMForecaster`). Splitting within the fold avoids ever needing a
    training-residual fallback, including for fold 0 (which has no previous fold to
    borrow a calibration set from).
    """
    split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
    fit_df = train_df.filter(pl.col("time") < split_point)
    calib_df = train_df.filter(pl.col("time") >= split_point)
    if fit_df.height < 200 or calib_df.height < 200:
        return None

    fit_nominal = filter_nominal_rows(fit_df, labels_long, target_channels)
    if fit_nominal.height < 200:
        return None

    model = PCAReconstructor(n_components=0.95).fit(channel_matrix(fit_nominal, target_channels))

    calib_values = channel_matrix(calib_df, target_channels)
    calib_y = row_is_non_nominal(calib_df, labels_long, target_channels)
    if calib_y.sum() > 0:
        tuned = find_best_threshold(model.score(calib_values), calib_y, beta=0.5)
        model.threshold_ = tuned["threshold"] if tuned["f_beta"] > 0 else float(np.percentile(model.score(calib_values), 99.0))
    else:
        model.threshold_ = float(np.percentile(model.score(calib_values), 99.0))
    return model


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta**2
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f_beta": f_beta, "beta": beta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--sliding-months", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None, help="bound the number of folds for a fast demo run")
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

    folds = list(
        generate_walk_forward_folds(config.split.test_start, config.split.test_end, step_months=args.step_months, window="expanding")
    )
    if args.max_folds:
        folds = folds[: args.max_folds]
    print(f"{len(folds)} walk-forward folds over the test period ({config.split.test_start.date()} .. {config.split.test_end.date()})")

    expanding_results, sliding_results, hybrid_results = [], [], []

    t0 = time.time()
    for fold in folds:
        train_df = _load_slice(lf, fold.train_start, fold.train_end, target_channels)
        eval_df = _load_slice(lf, fold.eval_start, fold.eval_end, target_channels)
        if train_df.height < 300 or eval_df.height < 200:
            continue

        eval_values = channel_matrix(eval_df, target_channels)
        eval_y = row_is_non_nominal(eval_df, labels_long, target_channels)

        expanding_model = _fit_pca(train_df, target_channels, labels_long)

        sliding_start = max(config.split.test_start, fold.train_end - relativedelta(months=args.sliding_months))
        sliding_train_df = _load_slice(lf, sliding_start, fold.train_end, target_channels) if sliding_start > fold.train_start else train_df
        sliding_model = _fit_pca(sliding_train_df, target_channels, labels_long)

        if expanding_model is None or sliding_model is None:
            continue

        expanding_pred = expanding_model.predict(eval_values)
        sliding_pred = sliding_model.predict(eval_values)
        hybrid_pred = PairedLearnerEnsemble(expanding_model, sliding_model).predict(eval_values)

        expanding_results.append(_metrics(eval_y, expanding_pred))
        sliding_results.append(_metrics(eval_y, sliding_pred))
        hybrid_results.append(_metrics(eval_y, hybrid_pred))

        print(
            f"fold {fold.fold_index}: {fold.eval_start.date()}..{fold.eval_end.date()} "
            f"(train {fold.train_start.date()}..{fold.train_end.date()}) "
            f"expanding_f0.5={expanding_results[-1]['f_beta']:.3f} "
            f"sliding_f0.5={sliding_results[-1]['f_beta']:.3f} "
            f"hybrid_f0.5={hybrid_results[-1]['f_beta']:.3f}"
        )

    print(f"\nTotal walk-forward time: {time.time() - t0:.1f}s")
    for variant, results in [
        ("WalkForward-expanding", expanding_results),
        ("WalkForward-sliding", sliding_results),
        ("WalkForward-hybrid", hybrid_results),
    ]:
        agg = aggregate_fold_results(results)
        print(f"\n=== {variant}: {agg['n_folds']} folds ===")
        print("  micro:", agg["micro"])
        print("  macro F0.5:", agg["macro"].get("f_beta"))
        run_id = log_experiment(
            model_name=variant,
            mission=args.mission,
            config={"step_months": args.step_months, "sliding_months": args.sliding_months, "base_model": "PCAReconstructor(n_components=0.95)"},
            metrics={"micro": agg["micro"], "macro": agg["macro"]},
            dataset_scope={"n_folds": agg["n_folds"], "test_start": str(config.split.test_start), "test_end": str(config.split.test_end)},
            notes=f"Walk-forward over {agg['n_folds']} monthly folds, sliding_months={args.sliding_months}",
            mirror_to_mlflow=args.mlflow,
        )
        print(f"  logged as [{run_id}]")


if __name__ == "__main__":
    main()
