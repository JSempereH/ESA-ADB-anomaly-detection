#!/usr/bin/env python3
"""Single entry point for the full L0/L1/L2 supervision ladder, all 12 models (the
original 9 plus PyOD-ECOD/PyOD-COPOD/PyOD-IForest), one mission, one protocol
(full-history walk-forward, "online" variant) -- idempotent:
before launching any batch it checks data/experiments/results.jsonl for an
already-logged run (same model + notes prefix) that already carries the
anomalies-only metric (evaluation.anomalies_only / benchmark._pooled_metrics) and
skips any model already covered, training only the ones still missing.

Ladder levels (see README's supervision ladder table):
  L0: calibration=event,  fit=label-filtered nominal rows (label-tuned threshold)
  L1: calibration=budget (alarm_budget=20), fit=label-filtered nominal rows
      (unsupervised threshold, still label-informed fit)
  L2: calibration=budget (alarm_budget in 5/10/20/50), fit=ALL rows (fully
      unsupervised -- labels touch nothing except the final scoring)

NOTE on scope: the 4 original ladder models (GlobalSTD, PCAReconstructor,
LaggedRidgeForecaster, NearestNominalWindow) already have L0/L1/L2 logged, but from
BEFORE both this script and the anomalies-only metric existed, and NOT under
full-history (their runs only cover the mission's test period, ~20 folds, not the
full ~41-fold mission-from-month-1 walk that the other 5 models' supervision scripts
use). This script does not try to distinguish "no anomaly info" from "different
protocol" -- it always retrains a model if it can't find an already-logged
full-history run WITH anomaly info at that exact level, so ALL 9 models end up
comparable under one consistent protocol. That means the first run of this script
retrains those 4 models from scratch even though older (non-full-history) ladder data
exists for them -- restrict via --models if you don't want that right now.

Matches by NOTES PREFIX (not exact string), so runs already logged by the older
mission2_supervision_L0/L1/L2.sh scripts (which used a "-- remaining models" suffix)
are recognized as already-done and skipped too.

Re-run-safe: interrupt it any time and re-run the same command -- already-logged
levels are skipped, only what's missing resumes.

Usage:
  python scripts/run_supervision_ladder.py --mission 2
  python scripts/run_supervision_ladder.py --mission 2 --dry-run
  python scripts/run_supervision_ladder.py --mission 2 --force
  python scripts/run_supervision_ladder.py --mission 2 --models GlobalSTD,PCAReconstructor
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import load_benchmark_data, run_walk_forward

ALL_MODELS = [
    "GlobalSTD", "PCAReconstructor", "LaggedRidgeForecaster", "NearestNominalWindow",
    "SubsystemPCAEnsemble", "WindowedIsolationForest", "HalfSpaceTrees", "Autoencoder", "LSTMForecaster",
    "PyOD-ECOD", "PyOD-COPOD", "PyOD-IForest",
]


def load_runs(results_path: Path, mission: int) -> list[dict]:
    if not results_path.exists():
        return []
    runs = []
    for line in results_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("mission") == mission:
                runs.append(r)
    return runs


def has_anomaly_info(run: dict) -> bool:
    m = run.get("metrics", {})
    return "pooled_event_wise_corrected_anomalies_only" in m or "event_wise_corrected_anomalies_only" in m


def already_done(runs: list[dict], model: str, notes_prefix: str) -> bool:
    return any(
        r.get("model_name") == f"{model}-online" and r.get("notes", "").startswith(notes_prefix) and has_anomaly_info(r)
        for r in runs
    )


def ladder_batches():
    """Yields (label, notes_prefix_to_match, tag_to_log, calibration, alarm_budget, fit_on_all_rows)."""
    yield (
        "L0", "Full-history supervision L0:",
        "Full-history supervision L0: label-tuned threshold, label-filtered fit",
        "event", 20.0, False,
    )
    yield (
        "L1", "Full-history supervision L1:",
        "Full-history supervision L1: unsupervised threshold (budget 20/mo), label-filtered fit",
        "budget", 20.0, False,
    )
    for b in (5, 10, 20, 50):
        prefix = f"Full-history supervision L2: fully unsupervised (budget {b}/mo"
        yield (
            f"L2 (budget={b}/mo)", prefix, f"{prefix}, unfiltered fit)",
            "budget", float(b), True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--results", default="data/experiments/results.jsonl")
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--dry-run", action="store_true", help="print what would run without training anything")
    parser.add_argument("--force", action="store_true", help="ignore existing logged runs, retrain everything")
    args = parser.parse_args()

    models = [m for m in args.models.split(",") if m]
    results_path = Path(args.results)
    data = None  # loaded lazily, once, on the first batch that actually needs training

    for label, notes_prefix, tag, calibration, alarm_budget, fit_on_all_rows in ladder_batches():
        runs = load_runs(results_path, args.mission)  # reloaded each batch: sees this script's own prior batches too
        missing = models if args.force else [m for m in models if not already_done(runs, m, notes_prefix)]
        done = [m for m in models if m not in missing]
        print(f"=== {label} ===")
        print(f"  already done: {done or '(none)'}")
        if not missing:
            print("  nothing to train -- skipping\n")
            continue
        print(f"  training: {missing}")
        if args.dry_run:
            print("  (dry-run, not launching)\n")
            continue

        if data is None:
            print(f"  loading mission {args.mission} data...")
            data = load_benchmark_data(args.mission, args.config_dir)

        run_walk_forward(
            data, missing,
            variants=("online",),
            calibration=calibration,
            alarm_budget=alarm_budget,
            fit_on_all_rows=fit_on_all_rows,
            full_history=True,
            tag=tag,
        )
        print()

    print(f"supervision ladder done -- next: python scripts/analyze_paper_stats.py --mission {args.mission}")


if __name__ == "__main__":
    main()
