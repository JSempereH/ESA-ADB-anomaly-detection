#!/usr/bin/env python3
"""L2 (fully unsupervised) full-history walk-forward comparison: Autoencoder vs.
MemStream, evaluated against the exact protocol used for this project's other L2
numbers (`run_supervision_ladder.py`'s L2 batches): online variant, fit on ALL rows (no
label filter), threshold calibrated by an unsupervised alarm budget, full-history folds
(walk from the mission's first month). Mirrors `run_loe_l2_walkforward.py`: trains each
model ONCE per fold and calibrates every `--budgets` value from that single pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import _calibrate, _load_slice, _pooled_metrics, _predict, _supervision_descriptor, load_benchmark_data
from esa_adb.evaluation import evaluate_event_wise_corrected
from esa_adb.experiments import log_experiment
from esa_adb.models.base import channel_matrix
from esa_adb.models.postprocess import PostprocessSpec
from esa_adb.models.runners import build_runners
from esa_adb.models.walk_forward import generate_walk_forward_folds
from esa_adb.preprocessing.labeling import row_is_non_nominal

MODELS = ["Autoencoder", "MemStream"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--budgets", default="5,10,20,50", help="comma-separated max alarms per 30 days")
    parser.add_argument("--max-alarm-time-frac", type=float, default=0.02)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/memstream_l2_walkforward.json"))
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [float(v) for v in args.budgets.split(",")]
    model_names = [m for m in args.models.split(",") if m]

    pp = PostprocessSpec()
    data = load_benchmark_data(args.mission, args.config_dir)
    runners = build_runners(model_names, len(data.input_channels), data.channel_groups, data.spec, ae_seed=args.seed)

    folds = list(
        generate_walk_forward_folds(
            data.config.split.train_start, data.config.split.test_end,
            step_months=args.step_months, window="sliding", sliding_window_months=args.step_months,
        )
    )
    if args.max_folds:
        folds = folds[: args.max_folds]
    print(f"L2 full-history walk-forward: {len(folds)} folds, models={model_names}, budgets={budgets}")

    records: dict[str, dict[float, list[dict]]] = {name: {b: [] for b in budgets} for name in runners}
    t_start = time.time()

    for fold in folds:
        train_df = _load_slice(data, fold.train_start, fold.train_end)
        eval_df = _load_slice(data, fold.eval_start, fold.eval_end)
        if train_df.height < 300 or eval_df.height < 200:
            continue
        split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
        fit_df = train_df.filter(pl.col("time") < split_point)
        calib_df = train_df.filter(pl.col("time") >= split_point)
        if fit_df.height < 300 or calib_df.height < 200:
            continue

        # L2: fit on ALL rows, no label filter -- the truly unsupervised production condition.
        fit_values = channel_matrix(fit_df, data.input_channels)
        calib_values = channel_matrix(calib_df, data.input_channels)
        eval_values = channel_matrix(eval_df, data.input_channels)
        calib_times = calib_df["time"].to_numpy()
        eval_times = eval_df["time"].to_numpy()
        eval_y = row_is_non_nominal(eval_df, data.labels_long, data.target_channels)

        for name, runner in runners.items():
            t0 = time.time()
            calib_s, eval_s = runner.online_scores(fit_values, calib_values, eval_values)
            line = []
            for budget in budgets:
                thr, alpha = _calibrate(calib_s, calib_times, None, data.events, pp, "budget", budget, args.max_alarm_time_frac)
                pred = _predict(eval_s, thr, alpha, pp)
                ewc = evaluate_event_wise_corrected(pred, eval_times, data.events, beta=0.5)
                records[name][budget].append({"pred": pred, "times": eval_times, "y": eval_y, "event_wise_corrected": ewc})
                line.append(f"b{budget:g}={ewc['f_beta']:.3f}({ewc['n_detections']}det)")
            print(f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) {name}: " + " ".join(line) + f" [{time.time() - t0:.1f}s]")

    rows = []
    summary: dict[str, dict[float, dict]] = {name: {} for name in runners}
    for name, runner in runners.items():
        for budget in budgets:
            fold_records = records[name][budget]
            if not fold_records:
                continue
            agg = _pooled_metrics(fold_records, data.events)
            summary[name][budget] = agg
            run_id = log_experiment(
                model_name=f"{name}-online",
                mission=args.mission,
                config={
                    "protocol": "walk-forward", "step": f"{args.step_months}mo", "calibration": "budget",
                    "alarm_budget": budget, "fit_on_all_rows": True, "full_history": True, "ae_seed": args.seed,
                    "supervision": _supervision_descriptor("budget", True),
                    "merge_gap_rows": pp.merge_gap_rows, "min_duration_rows": pp.min_duration_rows,
                    "smooth_alphas": [a for a in pp.smooth_alphas], "n_channels": len(data.target_channels),
                } | runner.run_config(),
                metrics=agg,
                dataset_scope={"n_folds": agg["n_folds"], "test_start": str(data.config.split.train_start), "test_end": str(data.config.split.test_end)},
                notes=args.tag or f"Full-history supervision L2: fully unsupervised (budget {budget:g}/mo, unfiltered fit) -- MemStream comparison script (shared per-fold training across budgets)",
                mirror_to_mlflow=args.mlflow,
            )
            ewc = agg["pooled_event_wise_corrected"]
            ewc_anom = agg["pooled_event_wise_corrected_anomalies_only"]
            rows.append({
                "model": name, "budget": budget, "run_id": run_id,
                "f_beta_all": ewc["f_beta"], "recall_all": ewc["recall"], "precision_all": ewc["precision"],
                "f_beta_anomalies": ewc_anom["f_beta"], "recall_anomalies": ewc_anom["recall"], "precision_anomalies": ewc_anom["precision"],
                "n_detections": ewc["n_detections"],
            })
            print(
                f"=== {name} budget={budget:g}/mo: all-events F0.5={ewc['f_beta']:.3f} "
                f"(P={ewc['precision']:.3f} R={ewc['recall']:.3f}) | anomalies-only F0.5={ewc_anom['f_beta']:.3f} "
                f"(P={ewc_anom['precision']:.3f} R={ewc_anom['recall']:.3f}) [{run_id}]"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": model_names, "budgets": budgets, "results": rows}, indent=2, default=str))
    print(f"Wrote {args.output}")
    print(f"total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
