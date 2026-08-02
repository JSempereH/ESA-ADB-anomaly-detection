#!/bin/sh
# Supervision L2 (fully unsupervised: alarm budget + unfiltered fit, labels only used
# to evaluate) for the 5 models that never got the L1/L2 treatment:
# SubsystemPCAEnsemble + the 4 heavy models. Budgets match what's already logged for
# the other 4 ladder models (5/10/20/50 alarms/month), so the ladder table stays
# directly comparable across all 9 models.
# --full-history: walk-forward starts at the mission's actual first month
# (config.split.train_start), never warm-started mid-mission -- every fold, always
# starting cold from month 1.
#
# Usage: sh scripts/mission2_supervision_L2.sh

set -eu
MODELS=SubsystemPCAEnsemble,WindowedIsolationForest,HalfSpaceTrees,Autoencoder,LSTMForecaster

for b in 5 10 20 50; do
  uv run scripts/run_benchmark.py --mission 2 --protocol walk-forward --full-history \
    --models $MODELS --variants online \
    --calibration budget --alarm-budget "$b" --fit-on-all-rows \
    --tag "Full-history supervision L2: fully unsupervised (budget $b/mo, unfiltered fit) -- remaining models"
done

echo "L2 done -- next: python scripts/analyze_paper_stats.py --mission 2, then update the supervision ladder table in README.md"
