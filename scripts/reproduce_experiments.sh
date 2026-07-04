#!/bin/sh
# Re-run every experiment logged in this project's data/experiments/results.jsonl, with
# the exact commands and flags used to produce the numbers documented in README.md.
#
# Prerequisites (not run by this script -- see README's "Usage" section):
#   1. scripts/download_raw_data.sh 2 && scripts/unzip_raw_data.sh
#   2. python scripts/ingest_raw.py --mission 2
#   3. python scripts/run_preprocessing.py --mission 2
# These freeze configs/mission2.yaml's grid/split once and are idempotent (ingestion
# skips itself if data/interim/Mission2/channel_ingest_meta.parquet already exists) --
# safe to re-run, but not repeated by this script since they take much longer than the
# experiments themselves and don't need to change between experiment re-runs.
#
# Determinism: PCAReconstructor/WindowedIsolationForest/SubsystemPCAEnsemble use fixed
# random_state=42 (sklearn); Autoencoder/LSTMForecaster use seed=42 by default (see their
# docstrings and tests/unit/test_reproducibility.py) -- re-running this script should
# reproduce the same results bit-for-bit given the same input data and package versions.
#
# The walk-forward steps below cover the FULL Mission2 test period (no --max-folds cap,
# ~20 monthly folds each) to match what's documented in README.md -- this makes the full
# script noticeably slower than earlier versions (HalfSpaceTrees' expanding-window
# variant alone takes ~30 min, since both its "refit" and "online" paths reprocess a
# growing fit sub-period every fold); pass --max-folds N to any individual command below
# for a faster, partial-coverage smoke run instead.
#
# Usage: sh scripts/reproduce_experiments.sh [mission_number]

set -eu
MISSION="${1:-2}"

echo "=== Step 4/2/3: PCAReconstructor, WindowedIsolationForest, SubsystemPCAEnsemble, Autoencoder ==="
python scripts/run_experiments.py --mission "$MISSION" --ae-epochs 30

echo ""
echo "=== Step 1: walk-forward, PCAReconstructor expanding/sliding/hybrid, 3-month sliding window ==="
python scripts/demo_walk_forward.py --mission "$MISSION" --sliding-months 3

echo ""
echo "=== Open next steps: HalfSpaceTrees (single-split + refit-vs-online walk-forward, expanding history) ==="
python scripts/demo_online_hst.py --mission "$MISSION"

echo ""
echo "=== Open next steps: window_features.py re-benchmark + stricter subsystem agreement ==="
python scripts/demo_features_and_strict_subsystems.py --mission "$MISSION"

echo ""
echo "=== refit vs. online vs. hybrid, 1-month folds, across all 6 single-split models ==="
python scripts/demo_full_mission_walk_forward.py --mission "$MISSION"

echo ""
echo "=== dynamic thresholds: static vs. rolling-percentile vs. POT/EVT, on top of 'online' training ==="
python scripts/demo_dynamic_threshold.py --mission "$MISSION"

if [ "$MISSION" = "2" ]; then
  echo ""
  echo "=== corrected event-wise metric: replication (lightweight subset) + fair full-set comparison ==="
  python scripts/demo_corrected_event_wise.py --mission 2
fi

echo ""
echo "=== GDN (Graph Deviation Network) single-split benchmark ==="
python scripts/demo_gdn.py --mission "$MISSION" --epochs 15

echo ""
echo "=== paper-comparable single split: all lightweight models, consistent metrics ==="
python scripts/demo_lightweight_paper_split.py --mission "$MISSION"

if [ "$MISSION" = "2" ]; then
  echo ""
  echo "=== explainability: which channel(s) explain each detection (Autoencoder) ==="
  python scripts/demo_explain_detections.py --mission 2

  echo ""
  echo "=== explainability deep dive: graph-vs-subsystems, cross-model agreement, TP-vs-FP, temporal drift ==="
  python scripts/demo_explainability_deep_dive.py --mission 2
fi

echo ""
echo "All experiments re-run and logged to data/experiments/results.jsonl"
echo "Query with: python -c \"from esa_adb.experiments import summarize_experiments; print(summarize_experiments())\""
