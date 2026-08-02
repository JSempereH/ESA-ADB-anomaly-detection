#!/bin/sh
# Supervision L1 (unsupervised threshold: alarm budget, still label-filtered fit) for
# the 5 models that never got the L1/L2 treatment: SubsystemPCAEnsemble + the 4 heavy
# models. Fills the gap in the supervision ladder table in README.md, which so far only
# covers GlobalSTD, PCAReconstructor, LaggedRidgeForecaster, NearestNominalWindow.
# --full-history: walk-forward starts at the mission's actual first month
# (config.split.train_start), never warm-started mid-mission -- every fold, always
# starting cold from month 1.
#
# Usage: sh scripts/mission2_supervision_L1.sh

set -eu
MODELS=SubsystemPCAEnsemble,WindowedIsolationForest,HalfSpaceTrees,Autoencoder,LSTMForecaster

uv run scripts/run_benchmark.py --mission 2 --protocol walk-forward --full-history \
  --models $MODELS --variants online \
  --calibration budget --alarm-budget 20 \
  --tag "Full-history supervision L1: unsupervised threshold (budget 20/mo), label-filtered fit -- remaining models"

echo "L1 done -- next: sh scripts/mission2_supervision_L2.sh"
