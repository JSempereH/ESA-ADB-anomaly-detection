#!/usr/bin/env python3
"""Cold-start curve: how much history does a fully unsupervised detector need?

The L2 walk-forward gives every model the whole past. A satellite launched last week has
one month, or one week. This asks, for each anomaly-bearing evaluation month, what
happens if the (fresh, never-seen-the-mission) detector has only the previous `k` months
of telemetry: it is updated sequentially on those months (2/3 of each, as in the engine),
calibrates its alarm-budget threshold on the last third of the most recent one, and
scores the next month. Same L2 rules everywhere: unfiltered fit rows, label-free
threshold, labels only for scoring.

All `k` are scored on the SAME months (those with >= max(k) months of history and at
least one real anomaly), so the curve is paired. "all history" is the full walk-forward
run read back from `results.jsonl` on the same months.

Usage: scripts/run_cold_start.py --models TCopulaDetector,KalmanLocalTrend --budget 20 \
    --history 1,2,3,6,12 --output data/experiments/cold_start_m2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import _calibrate, _load_slice, _predict, load_benchmark_data
from esa_adb.evaluation import anomalies_only, evaluate_event_wise_corrected
from esa_adb.models.base import channel_matrix
from esa_adb.models.postprocess import PostprocessSpec
from esa_adb.models.runners import build_runners
from esa_adb.models.walk_forward import generate_walk_forward_folds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", type=int, default=2, choices=[1, 2])
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--models", required=True)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--history", default="1,2,3,6,12", help="months of history available to the fresh detector")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def split_fold(data, fold):
    train_df = _load_slice(data, fold.train_start, fold.train_end)
    if train_df.height < 300:
        return None
    cut = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
    return train_df.filter(pl.col("time") < cut), train_df.filter(pl.col("time") >= cut)


def reference_per_fold(mission: int, model: str, budget: float) -> dict[int, float]:
    """Per-fold anomalies-only F0.5 of the full-history online run, from the log."""
    best = None
    for line in Path("data/experiments/results.jsonl").read_text().splitlines():
        x = json.loads(line)
        c, m = x["config"], x["metrics"]
        if (x["mission"], x["model_name"]) != (mission, f"{model}-online") or c.get("calibration") != "budget":
            continue
        if not c.get("fit_on_all_rows") or float(c.get("alarm_budget") or -1) != budget or m.get("n_folds", 0) < 40:
            continue
        if "f_beta_anomalies_only" not in m["folds"][0]:
            continue
        if best is None or x["timestamp"] > best["timestamp"]:
            best = x
    return {} if best is None else {f["fold_index"]: f["f_beta_anomalies_only"] for f in best["metrics"]["folds"]}


def main() -> None:
    args = parse_args()
    ks = [int(k) for k in args.history.split(",")]
    models = [m for m in args.models.split(",") if m]
    data = load_benchmark_data(args.mission, args.config_dir)
    pp = PostprocessSpec()
    anomaly_events = anomalies_only(data.events)
    folds = list(generate_walk_forward_folds(
        data.config.split.train_start, data.config.split.test_end, step_months=1, window="sliding", sliding_window_months=1,
    ))
    # months with >=1 real anomaly in the evaluation window, and enough history for the largest k
    eligible = []
    for f in folds:
        if f.fold_index < max(ks) - 1:
            continue
        ev = _load_slice(data, f.eval_start, f.eval_end)
        if ev.height < 200:
            continue
        n_anom = evaluate_event_wise_corrected(np.zeros(ev.height, bool), ev["time"].to_numpy(), anomaly_events, beta=0.5)["n_events"]
        if n_anom > 0:
            eligible.append(f)
    print(f"{len(eligible)} anomaly-bearing months with >= {max(ks)} months of history")

    cache: dict[int, tuple] = {}

    def fold_parts(idx):
        if idx not in cache:
            cache[idx] = split_fold(data, folds[idx])
        return cache[idx]

    results = []
    for model in models:
        ref = reference_per_fold(args.mission, model, args.budget)
        per_k: dict[int, list[dict]] = {k: [] for k in ks}
        for f in eligible:
            eval_df = _load_slice(data, f.eval_start, f.eval_end)
            eval_values = channel_matrix(eval_df, data.input_channels)
            eval_times = eval_df["time"].to_numpy()
            parts_f = fold_parts(f.fold_index)
            if parts_f is None:
                continue
            last_calib = parts_f[1]
            calib_values = channel_matrix(last_calib, data.input_channels)
            calib_times = last_calib["time"].to_numpy()
            for k in ks:
                runner = build_runners([model], len(data.input_channels), data.channel_groups, data.spec)[model]
                for g in range(f.fold_index - k + 1, f.fold_index + 1):
                    parts = fold_parts(g)
                    if parts is None:
                        continue
                    runner.update(channel_matrix(parts[0], data.input_channels))
                calib_s, eval_s = runner.score_rows(calib_values), runner.score_rows(eval_values)
                thr, alpha = _calibrate(calib_s, calib_times, None, data.events, pp, "budget", args.budget, 0.02)
                pred = _predict(eval_s, thr, alpha, pp)
                m = evaluate_event_wise_corrected(pred, eval_times, anomaly_events, beta=0.5)
                per_k[k].append({"fold": f.fold_index, "f": m["f_beta"], "recall": m["recall"], "n_det": m["n_detections"],
                                 "n_events": m["n_events"], "n_hit": m["n_events_detected"]})
            print(f"{model} fold {f.fold_index}: " + " ".join(f"k={k}:{per_k[k][-1]['f']:.3f}" for k in ks) + (f" all:{ref[f.fold_index]:.3f}" if f.fold_index in ref else ""), flush=True)
        for k in ks:
            rows = per_k[k]
            results.append({
                "model": model, "history_months": k, "n_folds": len(rows),
                "macro_f05": float(np.mean([r["f"] for r in rows])),
                "recall": float(sum(r["n_hit"] for r in rows) / max(sum(r["n_events"] for r in rows), 1)),
                "alarms_per_month": float(np.mean([r["n_det"] for r in rows])),
                "per_fold": rows,
            })
        if ref:
            vals = [ref[f.fold_index] for f in eligible if f.fold_index in ref]
            results.append({"model": model, "history_months": "all", "n_folds": len(vals), "macro_f05": float(np.mean(vals))})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mission": args.mission, "budget": args.budget, "results": results}, indent=2, default=str))
    print("\nmodel, history_months, macro F0.5, recall, alarms/month")
    for r in results:
        print(r["model"], r["history_months"], f"{r['macro_f05']:.3f}", f"{r.get('recall', float('nan')):.2f}", f"{r.get('alarms_per_month', float('nan')):.0f}")


if __name__ == "__main__":
    main()
