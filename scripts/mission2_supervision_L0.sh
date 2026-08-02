#!/bin/sh
# Supervision L0 (label-tuned threshold, label-filtered fit) for the 5 models that
# never got the L1/L2 treatment: SubsystemPCAEnsemble + the 4 heavy models.
# --full-history: walk-forward starts at the mission's actual first month
# (config.split.train_start), never warm-started mid-mission -- every fold, always
# starting cold from month 1.
#
# NOTE: this is ALREADY COVERED -- results.jsonl has full_history=true,
# calibration=event entries for all 9 models, including these 5 (see the two
# "Full-history walk-forward" runs already logged). Running this script would just
# duplicate that data under a new tag. Skip straight to L1.
#
# Usage: sh scripts/mission2_supervision_L0.sh

set -eu
MODELS=SubsystemPCAEnsemble,WindowedIsolationForest,HalfSpaceTrees,Autoencoder,LSTMForecaster

uv run scripts/run_benchmark.py --mission 2 --protocol walk-forward --full-history \
  --models $MODELS --variants online \
  --tag "Full-history supervision L0: label-tuned threshold, label-filtered fit -- remaining models"

echo "L0 done -- next: sh scripts/mission2_supervision_L1.sh"
