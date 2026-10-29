#!/usr/bin/env python3
"""L2 anomalies-only ensemble/overlap analysis -- the open question the README's own
"Anomalies only" section raises and flags as untried: do the top L2 single-model
detectors (by anomalies-only F0.5) catch the SAME real anomaly events, or different
ones? If different, an OR-of-alarms ensemble should recall more events than any single
model at the cost of more alarms; if the same, ensembling buys nothing.

Runs each of `--models` through the exact L2 protocol used for the "Anomalies only"
README table (online variant, fit on ALL rows, budget-calibrated threshold, full-history
walk-forward) at its OWN previously-found best budget (`MODEL_BUDGETS`, read off that
table), captures each fold's post-processed alarm boolean array via
`run_walk_forward(..., return_records=True)`, then:

1. Per-event (real anomalies only) detection table per model, via `evaluate_events`.
2. Pairwise Jaccard overlap between models' detected-event sets.
3. An OR-of-alarms ensemble (union of all models' post-processed predictions,
   re-merged with the same `PostprocessSpec`), scored the same way, reporting both its
   anomalies-only F0.5 AND its total alarm count -- the honest cost side of the
   recall gain, since a union of N models' budgets is not free.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import load_benchmark_data, run_walk_forward
from esa_adb.evaluation import anomalies_only, evaluate_event_wise_corrected, evaluate_events
from esa_adb.models.postprocess import PostprocessSpec

# Best L2 (full-history, fit-on-all-rows, budget-calibrated) alarm budget per model,
# read off the main README's "Anomalies only: the number that matters" table.
MODEL_BUDGETS = {
    "PyOD-ECOD": 10.0,
    "PyOD-IForest": 50.0,
    "LSTMForecaster": 10.0,
    "WindowedIsolationForest": 5.0,
    "PyOD-COPOD": 10.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--models", default=",".join(MODEL_BUDGETS))
    parser.add_argument("--max-alarm-time-frac", type=float, default=0.02)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/l2_ensemble_overlap.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = [m for m in args.models.split(",") if m]
    pp = PostprocessSpec()
    data = load_benchmark_data(args.mission, args.config_dir)
    anomaly_events = anomalies_only(data.events)
    print(f"L2 ensemble/overlap: {model_names} against {anomaly_events.height} real anomaly events")

    pooled: dict[str, dict] = {}
    t_start = time.time()
    for name in model_names:
        budget = MODEL_BUDGETS[name]
        print(f"\n--- {name} (budget={budget:g}/mo) ---")
        _, records = run_walk_forward(
            data, [name],
            variants=("online",),
            step_months=args.step_months,
            calibration="budget",
            pp=pp,
            max_folds=args.max_folds,
            alarm_budget=budget,
            max_alarm_time_fraction=args.max_alarm_time_frac,
            fit_on_all_rows=True,
            full_history=True,
            return_records=True,
            mlflow=False,
            tag="l2-ensemble-overlap",
        )
        fold_records = records[name]["online"]
        pred = np.concatenate([r["pred"] for r in fold_records])
        times = np.concatenate([r["times"] for r in fold_records])
        pooled[name] = {"pred": pred, "times": times, "budget": budget}

    # Every model walked the SAME folds (fold inclusion only depends on data, not the
    # model), so the pooled timelines must line up row-for-row -- verified, not assumed,
    # before any per-event comparison below is allowed to mean anything.
    ref_times = pooled[model_names[0]]["times"]
    for name in model_names[1:]:
        if not np.array_equal(pooled[name]["times"], ref_times):
            raise RuntimeError(f"{name}'s pooled timeline doesn't match {model_names[0]}'s -- folds diverged, overlap comparison would be invalid")

    detected_by_model: dict[str, set] = {}
    for name in model_names:
        detected_df = evaluate_events(pooled[name]["pred"], pooled[name]["times"], anomaly_events)
        detected_by_model[name] = set(detected_df.filter(detected_df["detected"])["ID"].to_list())
        print(f"{name}: detects {len(detected_by_model[name])}/{anomaly_events.height} real anomaly events (budget {pooled[name]['budget']:g}/mo)")

    print("\n--- pairwise overlap (Jaccard over detected real-anomaly events) ---")
    overlap_rows = []
    for a, b in combinations(model_names, 2):
        sa, sb = detected_by_model[a], detected_by_model[b]
        union = sa | sb
        jaccard = len(sa & sb) / len(union) if union else 0.0
        only_a, only_b = sa - sb, sb - sa
        print(f"{a} vs {b}: Jaccard={jaccard:.3f}, {a}-only={len(only_a)}, {b}-only={len(only_b)}, both={len(sa & sb)}")
        overlap_rows.append({"a": a, "b": b, "jaccard": jaccard, "a_only": len(only_a), "b_only": len(only_b), "both": len(sa & sb)})

    union_events = set().union(*detected_by_model.values())
    print(f"\nUnion of all {len(model_names)} models' detected events: {len(union_events)}/{anomaly_events.height} "
          f"(best single model: {max(len(s) for s in detected_by_model.values())})")

    ensemble_pred = pp.apply(np.logical_or.reduce([pooled[name]["pred"] for name in model_names]))
    ewc_all = evaluate_event_wise_corrected(ensemble_pred, ref_times, data.events, beta=0.5)
    ewc_anom = evaluate_event_wise_corrected(ensemble_pred, ref_times, anomaly_events, beta=0.5)
    total_individual_budget = sum(pooled[name]["budget"] for name in model_names)
    print(f"\n--- OR ensemble ({'+'.join(model_names)}) ---")
    print(f"anomalies-only: F0.5={ewc_anom['f_beta']:.3f} P={ewc_anom['precision']:.3f} R={ewc_anom['recall']:.3f} "
          f"({ewc_anom['n_events_detected']}/{ewc_anom['n_events']} events), {ewc_anom['n_detections']} total detections")
    print(f"all-events: F0.5={ewc_all['f_beta']:.3f} P={ewc_all['precision']:.3f} R={ewc_all['recall']:.3f}")
    print(f"cost: {ewc_all['n_detections']} alarms over the scored timeline vs. sum of individual budgets ({total_individual_budget:g}/mo nominal)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "models": model_names,
        "budgets": {n: MODEL_BUDGETS[n] for n in model_names},
        "detected_events_per_model": {n: sorted(detected_by_model[n]) for n in model_names},
        "pairwise_overlap": overlap_rows,
        "union_detected_events": sorted(union_events),
        "n_anomaly_events": anomaly_events.height,
        "ensemble_anomalies_only": {k: v for k, v in ewc_anom.items()},
        "ensemble_all_events": {k: v for k, v in ewc_all.items()},
    }, indent=2, default=str))
    print(f"\nWrote {args.output}")
    print(f"total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
