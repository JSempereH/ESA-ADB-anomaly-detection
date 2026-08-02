#!/usr/bin/env python3
"""Paper-grade statistical analysis over data/experiments/results.jsonl.

Reads logged runs (no re-training) and prints:
1. Paired per-fold Wilcoxon signed-rank tests: online vs refit, event-cal vs row-cal
   (needs runs with per-fold metrics -- the "Rigor rerun" runs).
2. Postprocess-constant robustness tables (merge_gap / min_duration sweeps).
3. The triad ablation 2x2: {event,row} calibration x {with,without} postprocessing.
4. Multi-seed spread for the Autoencoder-replay.
5. Cold-start: warmup-2 L2 runs vs their no-warmup counterparts.
6. Channel-aware vs global.
7. Real anomalies only (excl. rare nominal events/comm gaps/invalid segments, see
   evaluation.anomalies_only): every logged run that has the
   `*_anomalies_only` metric (added after ESA-ADB's own "results for anomalies alone",
   Supplementary 4.2 -- older runs predate it and are skipped here, not zeroed).

Usage: python scripts/analyze_paper_stats.py --mission 2 [--results data/experiments/results.jsonl]

`--mission` matters once more than one mission is logged: run tags/model names are
identical across missions, so without the filter the "latest run wins" rule would
silently mix missions.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.stats import wilcoxon


def load_runs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(runs: list[dict], model_name: str, note_prefix: str) -> dict | None:
    """Most recent run matching model_name exactly and notes prefix -- reruns of the
    same experiment supersede older logs."""
    matches = [r for r in runs if r.get("model_name") == model_name and r.get("notes", "").startswith(note_prefix)]
    return max(matches, key=lambda r: r["timestamp"]) if matches else None


def pooled_f(run: dict | None) -> float | None:
    if not run:
        return None
    return run["metrics"].get("pooled_event_wise_corrected", {}).get("f_beta")


def fold_f(run: dict | None) -> list[dict] | None:
    if not run or "folds" not in run.get("metrics", {}):
        return None
    return run["metrics"]["folds"]


def paired_test(name: str, a_run: dict | None, b_run: dict | None, a_label: str, b_label: str) -> None:
    a_folds, b_folds = fold_f(a_run), fold_f(b_run)
    if a_folds is None or b_folds is None or len(a_folds) != len(b_folds):
        print(f"  {name}: missing per-fold data ({a_label}: {a_folds is not None}, {b_label}: {b_folds is not None}) -- skipped")
        return
    # Exclude folds with no annotated events, same as _pooled_metrics' macro average
    # (benchmark.py) -- a quiet month scores a hard 0.0 by convention (0/0 is
    # undefined), which would otherwise treat "correctly stayed silent" the same as "missed
    # a real event" in the paired comparison. Older logged runs predate the `n_events`
    # field; default to "has events" so those still compare exactly as before.
    pairs = [(f, g) for f, g in zip(a_folds, b_folds) if f.get("n_events", 1) > 0 and g.get("n_events", 1) > 0]
    if not pairs:
        print(f"  {name}: no scored folds (all quiet) -- skipped")
        return
    a = np.array([f["f_beta"] for f, _ in pairs])
    b = np.array([g["f_beta"] for _, g in pairs])
    diff = a - b
    if np.allclose(diff, 0):
        print(f"  {name}: identical fold scores -- no test needed")
        return
    stat, p = wilcoxon(a, b)
    print(
        f"  {name}: {a_label} median={np.median(a):.3f} vs {b_label} median={np.median(b):.3f} | "
        f"median diff={np.median(diff):+.3f}, wins {int((diff > 0).sum())}/{len(diff)} folds, "
        f"Wilcoxon p={p:.2g}{' *' if p < 0.05 else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/experiments/results.jsonl")
    parser.add_argument("--mission", type=int, default=2)
    args = parser.parse_args()
    runs = [r for r in load_runs(Path(args.results)) if r.get("mission") == args.mission]
    print(f"Mission {args.mission}: {len(runs)} logged runs\n")
    models = ["GlobalSTD", "PCAReconstructor", "LaggedRidgeForecaster"]

    print("=" * 78)
    print("1. PAIRED PER-FOLD TESTS (Wilcoxon signed-rank, 20 monthly folds)")
    print("=" * 78)
    print("online vs refit (event-calibrated):")
    for m in models:
        paired_test(m, latest(runs, f"{m}-online", "Rigor rerun: event"), latest(runs, f"{m}-refit", "Rigor rerun: event"), "online", "refit")
    print("event-cal vs row-cal (online):")
    for m in models:
        paired_test(m, latest(runs, f"{m}-online", "Rigor rerun: event"), latest(runs, f"{m}-online", "Rigor rerun: row"), "event", "row")

    print()
    print("=" * 78)
    print("2. POSTPROCESS-CONSTANT ROBUSTNESS (GlobalSTD online, pooled corrected F0.5)")
    print("=" * 78)
    baseline = pooled_f(latest(runs, "GlobalSTD-online", "Rigor rerun: event")) or pooled_f(latest(runs, "GlobalSTD-online", "P0: cheap"))
    print(f"  baseline (gap=100, dur=10): {baseline:.3f}" if baseline else "  baseline missing")
    rows = defaultdict(dict)
    for r in runs:
        notes = r.get("notes", "")
        if r.get("model_name") == "GlobalSTD-online" and notes.startswith("Robustness:"):
            key = notes.split("Robustness: ")[1]
            rows[key] = pooled_f(r)
    for key in sorted(rows):
        print(f"  {key}: {rows[key]:.3f}")

    print()
    print("=" * 78)
    print("3. TRIAD ABLATION 2x2 (online, pooled corrected F0.5)")
    print("=" * 78)
    print(f"  {'model':26s} {'event+pp':>9s} {'event-pp':>9s} {'row+pp':>9s} {'row-pp':>9s}")
    for m in models:
        cells = [
            pooled_f(latest(runs, f"{m}-online", "Rigor rerun: event")),
            pooled_f(latest(runs, f"{m}-online", "Ablation: event-cal WITHOUT")),
            pooled_f(latest(runs, f"{m}-online", "Rigor rerun: row")),
            pooled_f(latest(runs, f"{m}-online", "Ablation: row-cal WITHOUT")),
        ]
        print(f"  {m:26s} " + " ".join(f"{c:9.3f}" if c is not None else f"{'--':>9s}" for c in cells))

    print()
    print("=" * 78)
    print("4. MULTI-SEED SPREAD (Autoencoder-replay online, pooled corrected F0.5)")
    print("=" * 78)
    seed_runs = [r for r in runs if r.get("model_name") == "Autoencoder-replay-online" and "pooled_event_wise_corrected" in r.get("metrics", {})]
    by_seed = {}
    for r in seed_runs:
        by_seed[r.get("config", {}).get("ae_seed", 42)] = pooled_f(r)
    if by_seed:
        vals = np.array(list(by_seed.values()))
        print(f"  seeds {sorted(by_seed)}: " + ", ".join(f"{s}={v:.3f}" for s, v in sorted(by_seed.items())))
        print(f"  mean={vals.mean():.3f}, range={vals.max() - vals.min():.3f}")

    print()
    print("=" * 78)
    print("5. COLD-START (L2 budget 10/mo: no warmup vs 2-fold warmup)")
    print("=" * 78)
    for m in ["GlobalSTD", "LaggedRidgeForecaster", "NearestNominalWindow"]:
        no_wu = pooled_f(latest(runs, f"{m}-online", "Supervision L2: fully unsupervised (budget 10/mo"))
        wu = pooled_f(latest(runs, f"{m}-online", "Cold-start: L2 budget 10/mo"))
        if no_wu is not None and wu is not None:
            print(f"  {m:26s} no-warmup={no_wu:.3f}  warmup-2={wu:.3f}  delta={wu - no_wu:+.3f}")
        else:
            print(f"  {m:26s} missing runs (no-warmup: {no_wu}, warmup: {wu})")

    print()
    print("=" * 78)
    print("6. CHANNEL-AWARE vs GLOBAL (online, corrected F0.5)")
    print("=" * 78)
    for m in models:
        ca = latest(runs, f"{m}-channel-aware", "Channel-aware")
        glob = pooled_f(latest(runs, f"{m}-online", "Rigor rerun: event")) or pooled_f(latest(runs, f"{m}-online", "P0: cheap"))
        if ca:
            met = ca["metrics"]
            print(
                f"  {m:26s} global={glob:.3f}  channel-macro={met['channel_aware_macro_f0.5']:.3f}  "
                f"subsystem-macro={met['subsystem_aware_macro_f0.5']:.3f}  "
                f"OR-global={met['global_or']['pooled_event_wise_corrected']['f_beta']:.3f}  "
                f"({met['n_channels_scored']}/{met['n_channels_total']} channels)"
            )
        else:
            print(f"  {m:26s} channel-aware run missing")

    print()
    print("=" * 78)
    print("7. REAL ANOMALIES ONLY -- are we catching the actual anomalies, not rare events?")
    print("=" * 78)
    anomaly_runs = []
    for r in runs:
        metrics = r.get("metrics", {})
        anom = metrics.get("pooled_event_wise_corrected_anomalies_only") or metrics.get("event_wise_corrected_anomalies_only")
        all_ewc = metrics.get("pooled_event_wise_corrected") or metrics.get("event_wise_corrected")
        if anom is not None and all_ewc is not None:
            anomaly_runs.append((r, all_ewc, anom))
    if not anomaly_runs:
        print("  no logged runs have the anomalies-only metric yet -- rerun any walk-forward/paper-split experiment to populate it")
    else:
        # Latest wins per (model_name, notes) -- reruns of the same experiment supersede.
        latest_by_key: dict[tuple[str, str], tuple] = {}
        for r, all_ewc, anom in anomaly_runs:
            key = (r["model_name"], r.get("notes", ""))
            if key not in latest_by_key or r["timestamp"] > latest_by_key[key][0]["timestamp"]:
                latest_by_key[key] = (r, all_ewc, anom)
        # `notes` are long and near-identical across budgets (only the "budget N/mo"
        # part differs) -- pull that out into its own column instead of truncating
        # notes and silently losing the one thing that distinguishes rows.
        import re

        def _short_notes(notes: str) -> str:
            budget = re.search(r"budget (\d+)/mo", notes)
            protocol = notes.split(":")[0].strip()
            return f"{protocol} (budget={budget.group(1)}/mo)" if budget else protocol

        def _budget_sort_key(notes: str) -> int:
            budget = re.search(r"budget (\d+)/mo", notes)
            return int(budget.group(1)) if budget else -1  # non-budget runs (e.g. event-cal) sort first

        print(f"  {'model':28s} {'run':45s} {'F0.5(all)':>10s} {'F0.5(anom)':>11s} {'recall(anom)':>13s} {'anomalies':>11s}")
        for model, notes in sorted(latest_by_key, key=lambda k: (k[0], _budget_sort_key(k[1]))):
            r, all_ewc, anom = latest_by_key[(model, notes)]
            print(
                f"  {model:28s} {_short_notes(notes):45s} {all_ewc['f_beta']:10.3f} {anom['f_beta']:11.3f} "
                f"{anom['recall']:13.3f} {anom['n_events_detected']:>4d}/{anom['n_events']:<6d}"
            )


if __name__ == "__main__":
    main()
