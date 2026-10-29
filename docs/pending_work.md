# Status of the L2 follow-up round (written before a machine reset)

Everything below is reproducible from the scripts in `scripts/`. Raw numbers are in
`data/experiments/results.jsonl` (append-only) and the `*_l2.json` / `cold_start_m2.json` files.
Regenerate the ranking with `python scripts/analyze_leaderboard.py` (writes `docs/leaderboard.md`).

## Done

- **Uncertainty-aware leaderboard** (`scripts/analyze_leaderboard.py` -> `docs/leaderboard.md`):
  paired fold bootstrap, mean over the 4 alarm budgets (no label-based budget selection),
  alarms/month, % of time in alarm, alarms per detected anomaly. Main finding: the 95% CIs are very wide
  (best model 0.152 [0.047, 0.279]); only the gap between the copulas and the bottom half of the table is
  clearly outside noise. TCopula vs Gaussian copula is a tie.
- **New detectors** (all tested in `tests/unit/test_new_detectors.py`): `VARMahalanobis`, `HotellingPCA`,
  `LodaDetector`, `EchoStateForecaster`, `ChronosResidualDetector` (zero-shot, needs `uv pip install
  chronos-forecasting`, not a project dependency), and the `CUSUM-<model>` wrapper. Full 4-budget L2 sweeps
  finished for the first four. None beat the copulas on real anomalies (mean over budgets 0.017-0.068).
- **Mission 1 out-of-domain check** of the two remaining top-5 models (budget 20/mo, anomalies-only macro F0.5):
  `OnlineIsolationForest` 0.110 -> 0.047, `RoCALite` 0.114 -> 0.020. Only Kalman, ECOD and
  HeterogeneousFusion hold up out of domain.
- **Cold-start curve** (`scripts/run_cold_start.py`, `data/experiments/cold_start_m2.json`, budget 20/mo,
  15 anomaly months that have >= 12 months of history): macro F0.5 with 1/2/3/6/12 months of history vs the
  full-history run. TCopula 0.118 at every length (it refits per fold, so history is irrelevant to it),
  KalmanLocalTrend 0.055-0.080 vs 0.069 full, ECOD 0.087-0.092 vs 0.086 full: more history does not help
  these. VARMahalanobis is noisy (0.048-0.127). Caveat: shortest tested history is 1 month, and the
  evaluated months are late in the mission; the first weeks after launch are NOT covered.

## Invalid result, do not cite

The first `CUSUM-*` wrapper standardized scores with an in-sample reference. The statistic ramped without bound
and produced one alarm covering 60-70% of the month; `CUSUM-TCopulaDetector` showed pooled F0.5 0.131 (best ever)
purely from that. Fixed (reference = previous out-of-sample chunk, regression test added) and
`analyze_leaderboard.py` skips any `CUSUM-*` row without `cusum_ref` in its config. Old rows stay in
`results.jsonl` (append-only).

Side finding worth documenting: the corrected event-wise F0.5 can be gamed by a detector that
keeps most of the timeline in alarm. Several current models already sit at 20-40% time-in-alarm (see the
`% time in alarm` column), so report it next to F0.5.

## Not finished

1. **Fixed CUSUM sweep incomplete**: budgets 5 and 10 done for all three bases; budget 20 done only for TCopula,
   budget 50 done for none. Rerun:
   `python scripts/run_l2_budget_sweep.py --models CUSUM-TCopulaDetector,CUSUM-KalmanLocalTrend,CUSUM-PyOD-ECOD --output data/experiments/cusum_fixed_l2.json`
   The CUSUM-ECOD / CUSUM-Kalman rows currently ahead of the copulas (0.22 / 0.17 over 2 budgets) still keep
   40-58% of the time in alarm, so they are probably the same degenerate regime. Do not trust them until
   the remaining budgets run and the time-in-alarm is understood.
2. **Chronos zero-shot sweep not run to completion** (was at fold 14/41, ~130 s/fold, budget 20 only). Rerun:
   `python scripts/run_l2_budget_sweep.py --models ChronosResidual --budgets 20 --output data/experiments/chronos_l2.json`
   (about 1.6 h; run alone, it needs ~1 GB RAM and does not mix well with parallel jobs).
3. **README not updated** with this round. It still has the old, superseded anomalies-only table; it should
   link to `docs/leaderboard.md` instead and pick up the bootstrap CIs, the Mission 1 numbers above, the
   cold-start result and the time-in-alarm caveat.
4. Not tried: telecommand-conditioned models, SPOT/DSPOT as an alternative to the alarm budget, a cold-start
   curve for history shorter than one month.
5. `results.jsonl` contains 3-fold smoke-test rows (tag `smoke: new detectors 3 folds`); the analysis scripts
   ignore them (they require >= 40 folds).
