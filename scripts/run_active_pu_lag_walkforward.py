#!/usr/bin/env python3
"""Lagged-ground-truth experiment: PU-lag and active-lag, both discussed with the user
as a realistic middle ground between L2 (zero labels ever) and this project's L1
(oracle label-filtered fit set, no causality/lag). See `esa_adb.models.active_lag`'s
module docstring for the exact protocol each variant assumes.

Both variants reuse the project's main walk-forward infrastructure (persisted online
state across folds, `calibration="budget"` alarm-budget thresholding) rather than
re-deriving a new protocol, so results are directly comparable to the existing L2
numbers for `Autoencoder` -- the only axis that changes is what `positive=`/
`hard_negative=` the model's `fit`/`partial_fit` receives each fold.

- `--mode pu`: PU-lag only (`revealed_oracle_positives`).
- `--mode active`: active-lag only (`AlarmHistory`, confirmed from the model's OWN
  past alarms).
- `--mode both`: PU-lag positives AND active-lag hard negatives together.
- A plain `Autoencoder` (no lag feedback at all) is always run alongside as the
  same-protocol baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import _calibrate, _load_slice, _pooled_metrics, _predict, load_benchmark_data
from esa_adb.evaluation import evaluate_event_wise_corrected
from esa_adb.experiments import log_experiment
from esa_adb.models.active_lag import AlarmHistory, revealed_oracle_positives
from esa_adb.models.base import channel_matrix
from esa_adb.models.heavy import Autoencoder
from esa_adb.models.heavy.pu_lag_autoencoder import PULagAutoencoder
from esa_adb.models.postprocess import PostprocessSpec
from esa_adb.models.walk_forward import generate_walk_forward_folds
from esa_adb.preprocessing.labeling import row_is_non_nominal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--mode", default="both", choices=["pu", "active", "both"])
    parser.add_argument("--lag-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--alarm-budget", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/active_pu_lag_l2.json"))
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pp = PostprocessSpec()
    data = load_benchmark_data(args.mission, args.config_dir)
    n_channels = len(data.input_channels)

    use_pu = args.mode in ("pu", "both")
    use_active = args.mode in ("active", "both")
    lag_model = PULagAutoencoder(n_channels=n_channels, seed=args.seed)
    baseline_model = Autoencoder(n_channels=n_channels, hidden_sizes=(32, 16), latent_size=8, seed=args.seed)
    history = AlarmHistory(lag_months=args.lag_months) if use_active else None

    folds = list(
        generate_walk_forward_folds(
            data.config.split.train_start, data.config.split.test_end,
            step_months=args.step_months, window="sliding", sliding_window_months=args.step_months,
        )
    )
    if args.max_folds:
        folds = folds[: args.max_folds]
    print(f"Active/PU-lag walk-forward: {len(folds)} folds, mode={args.mode}, lag={args.lag_months}mo")

    records: dict[str, list[dict]] = {"lag_model": [], "baseline": []}
    t_start = time.time()

    for fold in folds:
        train_df = _load_slice(data, fold.train_start, fold.train_end)
        eval_df = _load_slice(data, fold.eval_start, fold.eval_end)
        if train_df.height < 300 or eval_df.height < 200:
            continue
        split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
        fit_df = train_df.filter(train_df["time"] < split_point)
        calib_df = train_df.filter(train_df["time"] >= split_point)
        if fit_df.height < 300 or calib_df.height < 200:
            continue

        fit_values = channel_matrix(fit_df, data.input_channels)
        calib_values = channel_matrix(calib_df, data.input_channels)
        eval_values = channel_matrix(eval_df, data.input_channels)
        calib_times = calib_df["time"].to_numpy()
        eval_times = eval_df["time"].to_numpy()
        eval_y = row_is_non_nominal(eval_df, data.labels_long, data.target_channels)

        # --- baseline: plain blind Autoencoder, same protocol, no lag feedback ---
        fit_fn = baseline_model.fit if fold.fold_index == 0 else baseline_model.partial_fit
        fit_fn(fit_values, verbose=False)
        calib_s, eval_s = baseline_model.score(calib_values), baseline_model.score(eval_values)
        thr, alpha = _calibrate(calib_s, calib_times, None, data.events, pp, "budget", args.alarm_budget)
        pred_baseline = _predict(eval_s, thr, alpha, pp)

        # --- lag model: gather exemplars revealed as of this fold's fit start ---
        positive = revealed_oracle_positives(data, fold.train_start, args.lag_months) if use_pu else None
        hard_negative = None
        if use_active and history is not None:
            tp, fp = history.confirm_ready(data, fold.train_start)
            if tp is not None:
                positive = tp if positive is None else np.concatenate([positive, tp], axis=0)
            hard_negative = fp

        fit_fn = lag_model.fit if fold.fold_index == 0 else lag_model.partial_fit
        fit_fn(fit_values, positive=positive, hard_negative=hard_negative, verbose=False)
        calib_s, eval_s = lag_model.score(calib_values), lag_model.score(eval_values)
        thr, alpha = _calibrate(calib_s, calib_times, None, data.events, pp, "budget", args.alarm_budget)
        pred_lag = _predict(eval_s, thr, alpha, pp)

        if use_active and history is not None:
            history.record(fold.fold_index, fold.eval_start, fold.eval_end, eval_times, eval_values, pred_lag)

        for name, pred in [("lag_model", pred_lag), ("baseline", pred_baseline)]:
            ewc = evaluate_event_wise_corrected(pred, eval_times, data.events, beta=0.5)
            records[name].append({"pred": pred, "times": eval_times, "y": eval_y, "event_wise_corrected": ewc})

        n_pos = 0 if positive is None else len(positive)
        n_neg = 0 if hard_negative is None else len(hard_negative)
        print(
            f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) "
            f"lag_model(P={n_pos},FP={n_neg}): F0.5={records['lag_model'][-1]['event_wise_corrected']['f_beta']:.3f} "
            f"| baseline: F0.5={records['baseline'][-1]['event_wise_corrected']['f_beta']:.3f}"
        )

    rows = []
    for name in records:
        if not records[name]:
            continue
        agg = _pooled_metrics(records[name], data.events)
        run_id = log_experiment(
            model_name=f"PULagAutoencoder-{args.mode}-lag{args.lag_months}mo" if name == "lag_model" else "Autoencoder-same-protocol-baseline",
            mission=args.mission,
            config={
                "protocol": "walk-forward-online-budget-calibration", "mode": args.mode, "lag_months": args.lag_months,
                "fit_on_all_rows": True, "full_history": False, "seed": args.seed, "alarm_budget": args.alarm_budget,
                "supervision": "L1-lag: fit=all-rows (unfiltered); threshold=unsupervised (alarm budget); "
                               "positive/hard-negative exemplars use REAL labels revealed with a causal lag -- NOT L2, do not compare directly",
            },
            metrics=agg,
            dataset_scope={"n_folds": agg["n_folds"], "test_start": str(data.config.split.train_start), "test_end": str(data.config.split.test_end)},
            notes=args.tag or f"Lagged-ground-truth PU/active-learning experiment (mode={args.mode}, lag={args.lag_months}mo)",
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
    args.output.write_text(json.dumps({"mode": args.mode, "lag_months": args.lag_months, "results": rows}, indent=2, default=str))
    print(f"Wrote {args.output}")
    print(f"total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
