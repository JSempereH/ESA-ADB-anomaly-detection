#!/usr/bin/env python3
"""Replication attempt of arXiv:2609.10017 ("Structure-Aware Unsupervised Anomaly
Detection for Spacecraft Telemetry with Adaptive EVT Thresholding", Alcarria, Sanchez,
Sempere et al., SPAICE 2026) on this project's own Mission 2 data/pipeline.

Protocol differences from this project's usual walk-forward scripts, deliberately
matched to the paper rather than this project's own convention:
- Each fold trains EXCLUSIVELY on the previous month (no 2/3-fit/1/3-calibration split
  within the training window) and is evaluated on the next month.
- Calibration scores for the EVT threshold are the training month's own IN-SAMPLE
  scores (the paper never mentions a held-out calibration slice), not a separate split.
- The model refits from scratch every month (no persistent online state) -- this
  project's own "online never forgets" default is set aside here specifically because
  the paper's own protocol is a fresh monthly refit.
- Threshold: `EVTCalibrator` (q=0.98, alpha=0.01, min_excesses=30, lam=delta=0.05, the
  paper's own reported defaults), NOT this project's usual alarm-budget calibration.

Still NOT identical to the paper: fit is on ALL rows unfiltered here (this project's L2
condition) rather than the paper's own MAD-prefiltered fit set (that prefiltering is
applied INSIDE `ClusterRegimeDetector`/would need adding to `KalmanLocalTrend`/
`Autoencoder` separately to match exactly); and `Autoencoder`/`KalmanLocalTrend` are
included as this project's own reference points, not the paper's actual Mission-1 LSTM
autoencoder (which needs real Mission-1 data this project hasn't run yet).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import _load_slice, _pooled_metrics, load_benchmark_data
from esa_adb.evaluation import evaluate_event_wise_corrected
from esa_adb.experiments import log_experiment
from esa_adb.models.evt_calibration import EVTCalibrator
from esa_adb.models.base import channel_matrix
from esa_adb.models.heavy import Autoencoder
from esa_adb.models.lightweight import ClusterRegimeDetector, KalmanLocalTrend
from esa_adb.models.postprocess import PostprocessSpec, ewma
from esa_adb.models.walk_forward import generate_walk_forward_folds
from esa_adb.preprocessing.labeling import row_is_non_nominal

MODELS = ["ClusterRegimeDetector", "KalmanLocalTrend", "Autoencoder"]


def _model_factories(n_channels: int, seed: int) -> dict:
    """Zero-arg constructors per model name -- used directly here instead of going
    through `build_runners`/`ScoreRunner`, since this script's protocol (fresh refit
    every fold, in-sample EVT calibration, no fit/calibration split) doesn't match any
    existing runner's `online_scores` contract closely enough to reuse it as-is."""
    return {
        "ClusterRegimeDetector": lambda: ClusterRegimeDetector(seed=seed),
        "KalmanLocalTrend": lambda: KalmanLocalTrend(),
        "Autoencoder": lambda: Autoencoder(n_channels=n_channels, hidden_sizes=(32, 16), latent_size=8, seed=seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/sempere_replication_l2.json"))
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = [m for m in args.models.split(",") if m]

    pp = PostprocessSpec()
    data = load_benchmark_data(args.mission, args.config_dir)
    factories = _model_factories(len(data.input_channels), args.seed)
    factories = {name: factories[name] for name in model_names}
    calibrators = {name: EVTCalibrator() for name in factories}

    folds = list(
        generate_walk_forward_folds(
            data.config.split.train_start, data.config.split.test_end,
            step_months=args.step_months, window="sliding", sliding_window_months=args.step_months,
        )
    )
    if args.max_folds:
        folds = folds[: args.max_folds]
    print(f"Sempere-replication walk-forward: {len(folds)} folds, models={model_names}")

    records: dict[str, list[dict]] = {name: [] for name in factories}
    t_start = time.time()

    for fold in folds:
        train_df = _load_slice(data, fold.train_start, fold.train_end)
        eval_df = _load_slice(data, fold.eval_start, fold.eval_end)
        if train_df.height < 300 or eval_df.height < 200:
            continue

        # L2: fit on the whole previous month, unfiltered -- no fit/calibration split.
        fit_values = channel_matrix(train_df, data.input_channels)
        eval_values = channel_matrix(eval_df, data.input_channels)
        eval_times = eval_df["time"].to_numpy()
        eval_y = row_is_non_nominal(eval_df, data.labels_long, data.target_channels)

        for name, factory in factories.items():
            t0 = time.time()
            model = factory()
            try:
                model.fit(fit_values, verbose=False)
            except TypeError:
                model.fit(fit_values)
            calib_scores = model.score(fit_values)  # in-sample, as the paper's own text implies
            eval_scores = model.score(eval_values)

            threshold = calibrators[name].calibrate(calib_scores)
            pred = pp.apply(ewma(eval_scores, None) > threshold)
            ewc = evaluate_event_wise_corrected(pred, eval_times, data.events, beta=0.5)
            records[name].append({"pred": pred, "times": eval_times, "y": eval_y, "event_wise_corrected": ewc})
            print(f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) {name}: "
                  f"F0.5={ewc['f_beta']:.3f} ({ewc['n_detections']}det) thr={threshold:.4g} [{time.time() - t0:.1f}s]")

    rows = []
    for name in factories:
        fold_records = records[name]
        if not fold_records:
            continue
        agg = _pooled_metrics(fold_records, data.events)
        run_id = log_experiment(
            model_name=f"{name}-sempere-replication",
            mission=args.mission,
            config={
                "protocol": "walk-forward-monthly-refit", "calibration": "evt",
                "fit_on_all_rows": True, "full_history": True, "seed": args.seed,
                "evt_q": 0.98, "evt_alpha": 0.01, "evt_lambda": 0.05, "evt_delta": 0.05,
                "supervision": "fit=all-rows (no label filter); threshold=unsupervised (EVT, no labels); labels used ONLY for final evaluation",
                "n_channels": len(data.target_channels),
            },
            metrics=agg,
            dataset_scope={"n_folds": agg["n_folds"], "test_start": str(data.config.split.train_start), "test_end": str(data.config.split.test_end)},
            notes=args.tag or "Replication attempt of arXiv:2609.10017's protocol (monthly refit, EVT thresholding, fully label-free)",
            mirror_to_mlflow=args.mlflow,
        )
        ewc = agg["pooled_event_wise_corrected"]
        ewc_anom = agg["pooled_event_wise_corrected_anomalies_only"]
        rows.append({
            "model": name, "run_id": run_id,
            "f_beta_all": ewc["f_beta"], "recall_all": ewc["recall"], "precision_all": ewc["precision"],
            "f_beta_anomalies": ewc_anom["f_beta"], "recall_anomalies": ewc_anom["recall"], "precision_anomalies": ewc_anom["precision"],
            "macro_f0.5_all": agg["macro_event_wise_corrected_f0.5"],
            "macro_f0.5_anomalies_only": agg["macro_f0.5_anomalies_only"],
            "n_detections": ewc["n_detections"],
        })
        print(f"=== {name}: all-events F0.5={ewc['f_beta']:.3f} (P={ewc['precision']:.3f} R={ewc['recall']:.3f}) "
              f"macro={agg['macro_event_wise_corrected_f0.5']:.3f} | anomalies-only F0.5={ewc_anom['f_beta']:.3f} "
              f"(P={ewc_anom['precision']:.3f} R={ewc_anom['recall']:.3f}) macro={agg['macro_f0.5_anomalies_only']:.3f} [{run_id}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": model_names, "results": rows}, indent=2, default=str))
    print(f"Wrote {args.output}")
    print(f"total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
