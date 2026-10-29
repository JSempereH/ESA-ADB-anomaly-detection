#!/usr/bin/env python3
"""L2 sweep across alarm budgets for one or more models.

L2 means fully unsupervised: fit on unfiltered data, calibrate the threshold
from an alarm budget. Reuses benchmark.run_walk_forward directly, so results
are comparable to this project's other L2 result files.

Usage: scripts/run_l2_budget_sweep.py --models GBMForecaster,MatrixProfileDetector \
    --output data/experiments/gbm_mp_l2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import load_benchmark_data, run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--models", required=True, help="comma-separated model names (build_runners keys)")
    parser.add_argument("--budgets", default="5,10,20,50")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--full-history", action="store_true", default=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = [m for m in args.models.split(",") if m]
    budgets = [float(b) for b in args.budgets.split(",")]
    data = load_benchmark_data(args.mission, args.config_dir)

    rows = []
    for budget in budgets:
        summary = run_walk_forward(
            data, model_names, variants=("online",), calibration="budget", alarm_budget=budget,
            fit_on_all_rows=True, full_history=True, max_folds=args.max_folds,
            tag=args.tag or f"L2 budget sweep ({budget}/mo)", mlflow=args.mlflow,
        )
        for name in model_names:
            if "online" not in summary.get(name, {}):
                continue
            agg = summary[name]["online"]
            ewc = agg["pooled_event_wise_corrected"]
            ewc_anom = agg["pooled_event_wise_corrected_anomalies_only"]
            rows.append({
                "model": name, "budget": budget,
                "f_beta_all": ewc["f_beta"], "recall_all": ewc["recall"], "precision_all": ewc["precision"],
                "f_beta_anomalies": ewc_anom["f_beta"], "recall_anomalies": ewc_anom["recall"], "precision_anomalies": ewc_anom["precision"],
                "macro_f0.5_all": agg["macro_event_wise_corrected_f0.5"],
                "macro_f0.5_anomalies_only": agg["macro_f0.5_anomalies_only"],
                "n_detections": ewc["n_detections"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": model_names, "budgets": budgets, "results": rows}, indent=2, default=str))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
