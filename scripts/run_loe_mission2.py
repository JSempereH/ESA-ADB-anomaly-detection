#!/usr/bin/env python3
"""Compare blind Autoencoder, LOE-H, and LOE-S on contaminated ESA-ADB Mission 2.

The fixed chronological paper split is used in L2 mode: every training row is retained,
the validation threshold is selected by an unlabeled alarm budget, and labels are used
only after prediction. Each score stream is evaluated both against real anomalies alone
and against all annotated events, which includes rare nominal events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import _calibrate, _load_slice, _predict, load_benchmark_data
from esa_adb.evaluation import anomalies_only, evaluate_event_wise_corrected
from esa_adb.experiments import log_experiment
from esa_adb.models.base import channel_matrix
from esa_adb.models.postprocess import PostprocessSpec
from esa_adb.models.runners import build_runners


MODELS = ["Autoencoder", "LOEAutoencoder-H", "LOEAutoencoder-S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--budgets", default="5,10,20,50", help="comma-separated maximum alarms per 30 days")
    parser.add_argument("--epochs", type=int, default=20, help="identical training epochs for all models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/loe_mission2_l2.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [float(value) for value in args.budgets.split(",")]
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("--budgets must contain positive values")

    data = load_benchmark_data(2, args.config_dir)
    train_df = data.lf.filter(pl.col("split") == "train").select(["time", *data.target_channels]).collect()
    val_df = _load_slice(data, data.config.split.val_start, data.config.split.train_end)
    test_df = _load_slice(data, data.config.split.test_start, data.config.split.test_end)
    fit_values = channel_matrix(train_df, data.input_channels)
    val_values = channel_matrix(val_df, data.input_channels)
    test_values = channel_matrix(test_df, data.input_channels)
    val_times = val_df["time"].to_numpy()
    test_times = test_df["time"].to_numpy()
    del train_df, val_df, test_df

    print(f"Mission 2 L2 paper split: train={len(fit_values)}, validation={len(val_values)}, test={len(test_values)}")
    print("Training is unlabeled: all training rows are retained; labels are used only below for final metrics.")
    pp = PostprocessSpec()
    runners = build_runners(MODELS, len(data.input_channels), data.channel_groups, data.spec, ae_seed=args.seed)
    rows = []
    for name, runner in runners.items():
        # Architecture, optimizer, epochs, and seed match; only LOE's training loss differs.
        runner.refit_epochs = args.epochs
        validation_scores, test_scores = runner.refit_scores(fit_values, val_values, test_values)
        for budget in budgets:
            threshold, alpha = _calibrate(validation_scores, val_times, None, data.events, pp, "budget", alarm_budget=budget)
            prediction = _predict(test_scores, threshold, alpha, pp)
            all_events = evaluate_event_wise_corrected(prediction, test_times, data.events, beta=0.5)
            anomaly_events = evaluate_event_wise_corrected(prediction, test_times, anomalies_only(data.events), beta=0.5)
            metrics = {"all_events": all_events, "anomalies_only": anomaly_events, "threshold": threshold, "smooth_alpha": alpha}
            config = {
                "protocol": "paper-split-loe-comparison",
                "supervision": "fit=all rows; threshold=unsupervised alarm budget; labels used ONLY for final evaluation",
                "alarm_budget_per_30_days": budget,
                "epochs": args.epochs,
                "seed": args.seed,
                **runner.run_config(),
            }
            run_id = log_experiment(
                model_name=name,
                mission=2,
                config=config,
                metrics=metrics,
                dataset_scope={"train_rows": int(len(fit_values)), "test_rows": int(len(test_values))},
                notes="Matched blind-vs-LOE comparison; one score stream evaluated for anomalies-only and all-events.",
            )
            rows.append({"model": name, "budget": budget, "run_id": run_id, **metrics})
            print(
                f"{name:18s} budget={budget:4.0f}/mo | "
                f"anomalies F0.5={anomaly_events['f_beta']:.3f} R={anomaly_events['recall']:.3f} P={anomaly_events['precision']:.3f} | "
                f"all-events F0.5={all_events['f_beta']:.3f} R={all_events['recall']:.3f} P={all_events['precision']:.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": MODELS, "budgets": budgets, "results": rows}, indent=2, default=str))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
