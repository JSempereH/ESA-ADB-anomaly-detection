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

# ---------------------------------------------------------------------------------
# CURRENT methodology (README "Results"): the unified benchmark runner. Corrected
# event-wise F0.5 everywhere, event-wise-calibrated thresholds, alarm postprocessing.
# The legacy demo_* steps further down reproduce the OLDER numbers (window-level
# metrics, row-level-calibrated thresholds) kept in the README's history/appendix.
# ---------------------------------------------------------------------------------

echo "=== walk-forward benchmark: all models, refit/online/hybrid, monthly folds ==="
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,SubsystemPCAEnsemble --tag "P0: cheap models, event-calibrated + postprocessing"
python scripts/run_benchmark.py --mission "$MISSION" --models WindowedIsolationForest,HalfSpaceTrees,Autoencoder,LSTMForecaster --tag "P0: heavy models, event-calibrated + postprocessing"
python scripts/run_benchmark.py --mission "$MISSION" --models LaggedRidgeForecaster,NearestNominalWindow --tag "P3: new lightweight models, full walk-forward"

echo ""
echo "=== calibration A/B: legacy row-level threshold objective, same postprocessing ==="
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor --calibration row --tag "P0 A/B: legacy row-level calibration, same postprocessing"

echo ""
echo "=== paper-split benchmark: paper-comparable corrected event-wise numbers ==="
python scripts/run_benchmark.py --mission "$MISSION" --protocol paper-split --models GlobalSTD,PCAReconstructor,SubsystemPCAEnsemble,WindowedIsolationForest,HalfSpaceTrees,Autoencoder --tag "P0: paper-split, event-calibrated + postprocessing"
python scripts/run_benchmark.py --mission "$MISSION" --protocol paper-split --models LaggedRidgeForecaster,NearestNominalWindow --tag "P3: new lightweight models, paper split"

echo ""
echo "=== P1: Autoencoder online + replay buffer ==="
python scripts/run_benchmark.py --mission "$MISSION" --models Autoencoder-replay --variants online --ae-online-epochs 20 --ae-online-lr 0.0005 --tag "P1: AE online + replay 50k rows + 20 epochs/fold, lr 5e-4"

echo ""
echo "=== P2: retraining cadence -- fixed sweeps + drift-adaptive prequential ==="
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --step-weeks 1 --tag "P2: cadence sweep 1w"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --step-weeks 2 --tag "P2: cadence sweep 2w"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --step-months 2 --tag "P2: cadence sweep 2mo"
python scripts/run_cadence.py --mission "$MISSION" --model PCAReconstructor --tag "P2: adaptive cadence PCA"
python scripts/run_cadence.py --mission "$MISSION" --model GlobalSTD --tag "P2: adaptive cadence GlobalSTD"

echo ""
echo "=== Supervision ladder: how much do labels buy? (L0 = event-calibrated runs above) ==="
# L1: threshold chosen WITHOUT labels (alarm budget + time-fraction cap); fit still label-filtered.
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster,NearestNominalWindow --variants online --calibration budget --alarm-budget 20 --tag "Supervision L1: unsupervised threshold (budget 20/mo), label-filtered fit"
# L2: FULLY unsupervised -- labels only ever touch the final evaluation.
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster,NearestNominalWindow --variants online --calibration budget --alarm-budget 20 --fit-on-all-rows --tag "Supervision L2: fully unsupervised (budget 20/mo, unfiltered fit), labels only for evaluation"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster,NearestNominalWindow --variants online --calibration budget --alarm-budget 50 --fit-on-all-rows --tag "Supervision L2: fully unsupervised (budget 50/mo, unfiltered fit), labels only for evaluation"

echo ""
echo "=== P4: channel_15 exclusion + telecommand features A/Bs ==="
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor --variants online --exclude-channels channel_15 --tag "P4 A/B: without channel_15"
python scripts/build_telecommand_features.py --mission "$MISSION"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --with-telecommands --tag "P4 A/B: with telecommand features"

echo ""
echo "=== Channel-aware evaluation: per-channel thresholds/alarms (paper's hierarchy flavor) ==="
python scripts/run_benchmark.py --mission "$MISSION" --protocol channel-aware --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --tag "Channel-aware: per-channel thresholds, 6-fold calib buffer"

echo ""
echo "=== Postprocess-constant robustness (one-at-a-time around gap=100/dur=10 defaults) ==="
for gap in 0 25 50 200 400; do
  python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD --variants online --merge-gap-rows "$gap" --tag "Robustness: merge_gap=$gap (min_dur=10)"
done
for dur in 1 3 30 100; do
  python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD --variants online --min-duration-rows "$dur" --tag "Robustness: min_dur=$dur (merge_gap=100)"
done

echo ""
echo "=== Triad ablation + per-fold rigor reruns + multi-seed + cold-start warmup ==="
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --merge-gap-rows 0 --min-duration-rows 1 --smooth-alphas none --tag "Ablation: event-cal WITHOUT postprocessing"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --calibration row --merge-gap-rows 0 --min-duration-rows 1 --smooth-alphas none --tag "Ablation: row-cal WITHOUT postprocessing (legacy baseline)"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --tag "Rigor rerun: event-cal refit/online/hybrid with per-fold logging"
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,PCAReconstructor,LaggedRidgeForecaster --variants online --calibration row --tag "Rigor rerun: row-cal online with per-fold logging"
for seed in 7 123; do
  python scripts/run_benchmark.py --mission "$MISSION" --models Autoencoder-replay --variants online --ae-online-epochs 20 --ae-online-lr 0.0005 --ae-seed "$seed" --tag "Multi-seed: AE-replay online, seed=$seed"
done
python scripts/run_benchmark.py --mission "$MISSION" --models GlobalSTD,LaggedRidgeForecaster,NearestNominalWindow --variants online --calibration budget --alarm-budget 10 --fit-on-all-rows --warmup-folds 2 --tag "Cold-start: L2 budget 10/mo with 2-fold warmup"

echo ""
echo "=== Statistical analysis over everything above ==="
python scripts/analyze_paper_stats.py --mission "$MISSION"

# ---------------------------------------------------------------------------------
# LEGACY experiments (pre-run_benchmark methodology; window-level metrics,
# row-level-calibrated thresholds). Kept for provenance of the older logged results.
# ---------------------------------------------------------------------------------

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
