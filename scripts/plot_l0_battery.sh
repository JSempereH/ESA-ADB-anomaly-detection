#!/bin/sh
# Plot every fold-metrics run in the full-history L0 battery (Mission 2, all 9 models,
# calibration=event, refit/online/hybrid) -- 27 PNGs in data/experiments/plots/, one
# per model-variant, F0.5 + precision/recall + alarm volume vs. calendar time.
#
# Usage: sh scripts/plot_l0_battery.sh

set -eu
LIGHT="GlobalSTD LaggedRidgeForecaster NearestNominalWindow PCAReconstructor SubsystemPCAEnsemble"
HEAVY="Autoencoder HalfSpaceTrees LSTMForecaster WindowedIsolationForest"
LIGHT_NOTES="Full-history walk-forward: entire mission 2, no held-out splits"
HEAVY_NOTES="Full-history walk-forward: heavy models, entire mission 2, no held-out splits"

for m in $LIGHT; do
  for v in refit online hybrid; do
    uv run scripts/plot_fold_metrics.py --mission 2 --model-name "${m}-${v}" \
      --notes-prefix "$LIGHT_NOTES"
  done
done

for m in $HEAVY; do
  for v in refit online hybrid; do
    uv run scripts/plot_fold_metrics.py --mission 2 --model-name "${m}-${v}" \
      --notes-prefix "$HEAVY_NOTES"
  done
done

echo "done -- 27 PNGs in data/experiments/plots/"
