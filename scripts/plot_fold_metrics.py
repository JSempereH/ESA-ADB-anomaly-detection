#!/usr/bin/env python3
"""Plot a logged walk-forward run's per-fold metrics over calendar time.

Every run_benchmark.py walk-forward run logs `metrics.folds` -- one entry per fold with
eval_start/eval_end, corrected event-wise F0.5, precision, recall, n_detections,
n_events, n_events_detected (see esa_adb/benchmark.py::_pooled_metrics). A single pooled
scalar hides how a model behaves across time -- especially important for
`--full-history` runs, which span the WHOLE mission (commissioning months included)
rather than a short, roughly-uniform test window. This turns that per-fold list into a
PNG: F0.5, precision/recall and alarm volume vs. real calendar dates, with the mission's
own train/test split boundary marked so a full-history run's overlap with the existing
(test-period-only) runs is visually obvious.

Usage:
  python scripts/plot_fold_metrics.py --mission 2 --model-name GlobalSTD-online \
      --notes-prefix "Full-history walk-forward"
  python scripts/plot_fold_metrics.py --mission 2 --run-id a1b2c3d4e5f6
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esa_adb.config import load_mission_config


def load_runs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_run(runs: list[dict], run_id: str | None, model_name: str | None, notes_prefix: str) -> dict:
    if run_id:
        matches = [r for r in runs if r["run_id"] == run_id]
        if not matches:
            raise SystemExit(f"no run with run_id={run_id}")
        return matches[0]
    matches = [r for r in runs if r.get("model_name") == model_name and r.get("notes", "").startswith(notes_prefix)]
    if not matches:
        raise SystemExit(f"no run matching model_name={model_name!r} notes_prefix={notes_prefix!r}")
    return max(matches, key=lambda r: r["timestamp"])  # latest rerun wins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="data/experiments/results.jsonl")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--run-id", default=None, help="exact run to plot -- overrides --model-name/--notes-prefix")
    parser.add_argument("--model-name", default=None, help="e.g. GlobalSTD-online (required unless --run-id given)")
    parser.add_argument("--notes-prefix", default="", help="disambiguate reruns of the same model_name; picks the latest match")
    parser.add_argument("--out", default=None, help="output PNG path (default: data/experiments/plots/<run_id>_<model>.png)")
    args = parser.parse_args()

    if not args.run_id and not args.model_name:
        raise SystemExit("need --run-id or --model-name")

    runs = [r for r in load_runs(Path(args.results)) if r.get("mission") == args.mission]
    run = find_run(runs, args.run_id, args.model_name, args.notes_prefix)
    folds = run.get("metrics", {}).get("folds")
    if not folds:
        raise SystemExit(f"run {run['run_id']} ({run['model_name']}) has no per-fold metrics logged")

    folds = sorted(folds, key=lambda f: f["eval_start"])
    dates = [datetime.fromisoformat(f["eval_start"]) for f in folds]
    f_beta = [f["f_beta"] for f in folds]
    precision = [f["precision"] for f in folds]
    recall = [f["recall"] for f in folds]
    n_det = [f["n_detections"] for f in folds]
    n_events = [f["n_events"] for f in folds]
    # "anomalies_only" fields are absent from runs logged before this metric existed --
    # fall back to None so old results.jsonl entries still plot (just without that line).
    has_anomaly_fields = all("recall_anomalies_only" in f for f in folds)
    recall_anom = [f["recall_anomalies_only"] for f in folds] if has_anomaly_fields else None
    n_anom_events = [f["n_events_anomalies_only"] for f in folds] if has_anomaly_fields else None
    n_anom_detected = [f["n_events_detected_anomalies_only"] for f in folds] if has_anomaly_fields else None

    test_start = None
    try:
        config = load_mission_config(Path(args.config_dir) / f"mission{args.mission}.yaml")
        test_start = config.split.test_start
    except Exception:
        pass

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(dates, f_beta, marker="o", markersize=3, color="tab:blue")
    axes[0].set_ylabel("corrected event-wise F0.5")
    axes[0].set_title(f"{run['model_name']} (mission {args.mission}) -- {run.get('notes', '')[:70]}")
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].plot(dates, precision, marker="o", markersize=3, label="precision", color="tab:green")
    axes[1].plot(dates, recall, marker="o", markersize=3, label="recall (all events)", color="tab:orange")
    if recall_anom is not None:
        # Only real anomalies -- the paper's own "results for anomalies alone"
        # (evaluation.anomalies_only). NaN'd out on folds with zero real anomalies so a
        # quiet-of-anomalies month doesn't plot as a misleading recall=0 dip.
        recall_anom_masked = [r if n > 0 else float("nan") for r, n in zip(recall_anom, n_anom_events)]
        axes[1].plot(
            dates, recall_anom_masked, marker="D", markersize=5, linestyle="--",
            label="recall (real anomalies only)", color="tab:red",
        )
    axes[1].set_ylabel("precision / recall")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].bar(dates, n_det, width=20, label="alarms raised", color="tab:red", alpha=0.6)
    axes[2].bar(dates, n_events, width=10, label="annotated events (incl. rare events)", color="tab:gray", alpha=0.8)
    if n_anom_events is not None:
        axes[2].bar(dates, n_anom_events, width=6, label="real anomalies", color="black", alpha=0.9)
    axes[2].set_ylabel("count per fold")
    axes[2].set_xlabel("fold eval month")
    axes[2].legend(loc="upper right", fontsize=8)

    if test_start is not None and dates[0] < test_start < dates[-1]:
        for ax in axes:
            ax.axvline(test_start, color="black", linestyle="--", linewidth=1)
        axes[0].text(test_start, 1.03, "old train/test split", fontsize=8, ha="center", va="bottom")

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path("data/experiments/plots") / f"{run['run_id']}_{run['model_name']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path} ({len(folds)} folds, {dates[0].date()}..{dates[-1].date()})")
    if n_anom_events is not None:
        total_anom = sum(n_anom_events)
        total_detected = sum(n_anom_detected)
        print(f"real anomalies: {total_detected}/{total_anom} detected across the plotted period ({total_detected / total_anom:.1%})" if total_anom else "real anomalies: none in the plotted period")


if __name__ == "__main__":
    main()
