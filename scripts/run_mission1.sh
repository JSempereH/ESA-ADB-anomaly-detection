#!/bin/sh
# Staged Mission 1 launcher. Everything Mission 2 went through, sized for Mission 1's
# scale (14 years @ 30s = 14.7M rows, 58 target channels, 7-year test period = ~84
# monthly walk-forward folds vs Mission 2's 20).
#
# Stages (run them one at a time, in order; each is safe to re-run):
#   prep      ~1-2h   ingest raw pickles -> Parquet, preprocess, telecommand features
#   core      ~7-9h   everything the paper needs: full walk-forward benchmark,
#                     paper-split, supervision ladder, rigor/ablation, AE-replay
#   extended  ~6-8h   cadence sweeps + drift-adaptive, channel-aware, robustness grid,
#                     multi-seed, telecommand A/B
#   analyze   ~1min   statistical report over everything logged for Mission 1
#
# Usage:
#   sh scripts/run_mission1.sh prep
#   nohup sh scripts/run_mission1.sh core > mission1_core.log 2>&1 &     # overnight
#   nohup sh scripts/run_mission1.sh extended > mission1_ext.log 2>&1 &  # second night
#   sh scripts/run_mission1.sh analyze
#
# Notes:
# - Raw data must already be downloaded/unzipped (scripts/download_raw_data.sh 1 &&
#   scripts/unzip_raw_data.sh); prep checks and refuses otherwise.
# - Every run logs to data/experiments/results.jsonl with mission=1 -- Mission 2
#   results are never touched, and analyze filters by mission.
# - 14 real mission years may contain long telemetry gaps; the walk-forward engine
#   skips folds with too little data (train<300 or eval<200 rows) and reports how many
#   folds actually scored (`n_folds` in the log/summary). If many folds are skipped,
#   inspect the gap locations before interpreting pooled numbers.

set -eu
STAGE="${1:?usage: sh scripts/run_mission1.sh prep|core|extended|analyze}"
M=1
TRIO=GlobalSTD,PCAReconstructor,LaggedRidgeForecaster
LADDER_MODELS=GlobalSTD,PCAReconstructor,LaggedRidgeForecaster,NearestNominalWindow

case "$STAGE" in
prep)
  [ -d data/raw/ESA-Mission1/channels ] || { echo "ERROR: raw Mission1 data missing -- run scripts/download_raw_data.sh 1 && scripts/unzip_raw_data.sh first"; exit 1; }
  python scripts/ingest_raw.py --mission $M
  python scripts/run_preprocessing.py --mission $M
  python scripts/build_telecommand_features.py --mission $M
  echo "prep done -- next: sh scripts/run_mission1.sh core (consider nohup, ~7-9h)"
  ;;

core)
  echo "=== [1/8] walk-forward: cheap models, refit/online/hybrid ==="
  python scripts/run_benchmark.py --mission $M --models GlobalSTD,PCAReconstructor,SubsystemPCAEnsemble --tag "M1 core: cheap models, event-calibrated"
  echo "=== [2/8] walk-forward: new lightweight models ==="
  python scripts/run_benchmark.py --mission $M --models LaggedRidgeForecaster,NearestNominalWindow --tag "M1 core: new lightweight models"
  echo "=== [3/8] walk-forward: heavy models (longest step) ==="
  python scripts/run_benchmark.py --mission $M --models WindowedIsolationForest,HalfSpaceTrees,Autoencoder,LSTMForecaster --tag "M1 core: heavy models"
  echo "=== [4/8] P1: Autoencoder-replay ==="
  python scripts/run_benchmark.py --mission $M --models Autoencoder-replay --variants online --ae-online-epochs 20 --ae-online-lr 0.0005 --tag "M1 core: AE-replay online"
  echo "=== [5/8] paper-split ==="
  python scripts/run_benchmark.py --mission $M --protocol paper-split --models GlobalSTD,PCAReconstructor,SubsystemPCAEnsemble,WindowedIsolationForest,HalfSpaceTrees,Autoencoder,LaggedRidgeForecaster,NearestNominalWindow --tag "M1 core: paper-split"
  echo "=== [6/8] supervision ladder (L1 + L2 budgets 5/10/20) ==="
  python scripts/run_benchmark.py --mission $M --models $LADDER_MODELS --variants online --calibration budget --alarm-budget 20 --tag "Supervision L1: unsupervised threshold (budget 20/mo), label-filtered fit"
  for b in 5 10 20; do
    python scripts/run_benchmark.py --mission $M --models $LADDER_MODELS --variants online --calibration budget --alarm-budget $b --fit-on-all-rows --tag "Supervision L2: fully unsupervised (budget $b/mo, unfiltered fit), labels only for evaluation"
  done
  echo "=== [7/8] rigor reruns (per-fold logging for paired tests) ==="
  python scripts/run_benchmark.py --mission $M --models $TRIO --tag "Rigor rerun: event-cal refit/online/hybrid with per-fold logging"
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --calibration row --tag "Rigor rerun: row-cal online with per-fold logging"
  echo "=== [8/8] triad ablation (no-postprocess cells) ==="
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --merge-gap-rows 0 --min-duration-rows 1 --smooth-alphas none --tag "Ablation: event-cal WITHOUT postprocessing"
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --calibration row --merge-gap-rows 0 --min-duration-rows 1 --smooth-alphas none --tag "Ablation: row-cal WITHOUT postprocessing (legacy baseline)"
  python scripts/analyze_paper_stats.py --mission $M
  echo "core done -- next: sh scripts/run_mission1.sh extended (optional, ~6-8h)"
  ;;

extended)
  echo "=== [1/6] cadence sweep (1w / 2w / 2mo; 1mo already covered by core) ==="
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --step-weeks 1 --tag "P2: cadence sweep 1w"
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --step-weeks 2 --tag "P2: cadence sweep 2w"
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --step-months 2 --tag "P2: cadence sweep 2mo"
  echo "=== [2/6] drift-adaptive prequential cadence ==="
  python scripts/run_cadence.py --mission $M --model PCAReconstructor --tag "P2: adaptive cadence PCA"
  python scripts/run_cadence.py --mission $M --model GlobalSTD --tag "P2: adaptive cadence GlobalSTD"
  echo "=== [3/6] channel-aware evaluation (slowest block: ~84 folds x 58 channels) ==="
  python scripts/run_benchmark.py --mission $M --protocol channel-aware --models $TRIO --tag "Channel-aware: per-channel thresholds, 6-fold calib buffer"
  echo "=== [4/6] postprocess-constant robustness ==="
  for gap in 0 25 50 200 400; do
    python scripts/run_benchmark.py --mission $M --models GlobalSTD --variants online --merge-gap-rows $gap --tag "Robustness: merge_gap=$gap (min_dur=10)"
  done
  for dur in 1 3 30 100; do
    python scripts/run_benchmark.py --mission $M --models GlobalSTD --variants online --min-duration-rows $dur --tag "Robustness: min_dur=$dur (merge_gap=100)"
  done
  echo "=== [5/6] multi-seed AE-replay ==="
  for seed in 7 123; do
    python scripts/run_benchmark.py --mission $M --models Autoencoder-replay --variants online --ae-online-epochs 20 --ae-online-lr 0.0005 --ae-seed $seed --tag "Multi-seed: AE-replay online, seed=$seed"
  done
  echo "=== [6/6] telecommand features A/B ==="
  python scripts/run_benchmark.py --mission $M --models $TRIO --variants online --with-telecommands --tag "P4 A/B: with telecommand features"
  python scripts/analyze_paper_stats.py --mission $M
  echo "extended done"
  ;;

analyze)
  python scripts/analyze_paper_stats.py --mission $M
  ;;

*)
  echo "unknown stage '$STAGE' -- use prep|core|extended|analyze"
  exit 1
  ;;
esac
