# ESA-ADB anomaly detection

Lightweight, continuously trained anomaly detection on real satellite telemetry, built
on the [ESA Anomaly Detection Benchmark](https://zenodo.org/records/15237121) (Kotowski
et al., [arXiv:2406.17826](https://arxiv.org/abs/2406.17826)). Instead of the paper's
fit-once-on-half-the-mission protocol, models here walk forward through the mission and
update as telemetry arrives, then get scored with the paper's own **corrected
event-wise F0.5**, a metric built to reward few well-placed alarms rather than flagging
every row in sight. Everything runs on a laptop CPU: chunked Parquet, no GPU. Mission 2
is fully processed; Mission 1 is staged but not yet run.

See [`docs/literature_review.md`](docs/literature_review.md) for how this fits the
online/streaming TSAD literature.

> **Note:** The headline F0.5 scores every
> annotated event together, and in Mission 2 that pool is 10,981 "rare nominal event"
> instances for every 186 real anomalies (59:1). A model can score 0.97 by nailing the
> rare-but-normal patterns and still may miss real anomalies, I'm working on that 😵‍💫. See [Anomalies only](#anomalies-only-the-number-that-matters)


## Setup & pipeline

```bash
uv venv .venv && uv sync --extra dev --extra heavy && source .venv/bin/activate
pytest tests/
scripts/download_raw_data.sh 2 && scripts/unzip_raw_data.sh
python scripts/ingest_raw.py --mission 2
python scripts/run_preprocessing.py --mission 2          # ZOH-resample, label, stats
python scripts/build_telecommand_features.py --mission 2 # optional: TC activity features
```

`scripts/plot_fold_metrics.py` needs `matplotlib` (`--extra viz`); the PyOD wrappers
need `pyod`/`numba` (`--extra pyod`). List every extra you want together: `uv sync`
makes the venv match exactly what you pass, so a bare `uv sync --extra viz` on an
existing venv drops `torch` and `pytest` again.

## Methodology

`scripts/run_benchmark.py` runs every model through one engine
(`src/esa_adb/benchmark.py`): same calibration, same alarm postprocessing, directly
comparable numbers.

**Protocols**
- **walk-forward** (primary): monthly (or N-week) folds. Each fold trains on 2/3 of the
  month, calibrates on the rest, and scores the next, fully held-out month. Variants:
  `refit` (fresh model per fold), `online` (one model, `partial_fit` every fold, never
  forgets), `hybrid` (AND of both).
- `--full-history`: starts the walk at the mission's actual first month instead of the
  paper's held-out test period, so `online` is never warm-started mid-mission. Roughly
  doubles `n_folds`.
- **paper-split**: the paper's own fixed train/val/test split, for paper-comparable
  numbers.

**Three precision-oriented pieces, all ablatable:**
1. **Metric**: corrected event-wise F0.5 (validated against the paper's published
   0.949; we reproduce 0.938 on the same subset). Contiguous alarms collapse to one
   detection, and precision is multiplied by the time-domain true-negative rate, so
   flooding the timeline gets punished.
2. **Threshold calibration against that same metric**
   (`find_best_threshold_event_wise`). The legacy row-level F0.5 objective is kept as
   `--calibration row` for comparison.
3. **Alarm postprocessing** (`models/postprocess.py`): optional smoothing, merging
   alarms under 30 minutes apart, dropping alarms under 3 minutes. Redundant once the
   threshold is event-wise calibrated (see Ablations below), but load-bearing under the
   unsupervised budget mode.

**Supervision levels.** Every logged run records which labels it used
(`config.supervision`); the models themselves never train on labels.

| Level | Fit data | Threshold |
|---|---|---|
| **L0** | label-filtered nominal rows | label-tuned (best F0.5 on calibration events) |
| **L1** | label-filtered nominal rows | unsupervised: alarm budget (max N alarms / 30 days) |
| **L2** (production) | all rows, unfiltered | unsupervised, same as L1 |

L2 is the deployable configuration: labels touch nothing except the final score. It's
what a satellite with no anomaly history actually gets on day one.

`scripts/run_supervision_ladder.py --mission 2` runs the full L0/L1/L2 sweep (budgets
5/10/20/50) for all 12 models under `--full-history`. It's idempotent: rerun it any
time and it only trains what's still missing from `results.jsonl`.

## Results

### Anomalies only: the number that matters

Filter to `Category == "anomaly"` (`evaluation.anomalies_only`) and every model
collapses. This is L2, full-history: fully online, fully unsupervised, labels used only
to score (the L0 column here is also full-history, so it won't quite match the
non-full-history numbers further down).

| Model | F0.5, all events (L0) | F0.5, real anomalies (best L2) | budget | recall | precision |
|---|---|---|---|---|---|
| `PyOD-ECOD` | 0.391 | **0.104** | 10/mo | 0.11 | 0.024 |
| `PyOD-IForest` | 0.314 | 0.098 | 50/mo | 0.19 | 0.049 |
| `LSTMForecaster` | 0.066 | 0.093 | 10/mo | 0.25 | 0.017 |
| `WindowedIsolationForest` | 0.339 | 0.092 | 5/mo | 0.10 | 0.049 |
| `PyOD-COPOD` | 0.260 | 0.091 | 10/mo | 0.11 | 0.049 |
| `SubsystemPCAEnsemble` | 0.894 | 0.083 | 10/mo | 0.27 | 0.041 |
| `GlobalSTD` | 0.897 | 0.056 | 20/mo | 0.23 | 0.041 |
| `Autoencoder` | 0.669 | 0.052 | 50/mo | 0.39 | 0.011 |
| `NearestNominalWindow` | 0.765 | 0.047 | 20/mo | 0.42 | 0.014 |
| `LaggedRidgeForecaster` | 0.866 | 0.038 | 50/mo | 0.39 | 0.011 |
| `PCAReconstructor` | 0.858 | 0.023 | 20/mo | 0.32 | 0.007 |
| `HalfSpaceTrees` | 0.069 | 0.012 | 50/mo | 0.13 | 0.016 |

The ranking flips. `GlobalSTD` and `LaggedRidgeForecaster` top the all-events
leaderboard by nailing rare nominal events, then land near the bottom here. `PyOD-ECOD`,
near the bottom of the all-events table, comes out on top. Precision tops out at 4.9%:
of every 100 alarms these models raise, fewer than 5 point at a real anomaly. This is not a deployable detector (yet 🙂). AFAIK, nobody has yet tried calibrating
specifically against `anomalies_only`, checking overlap between the top models (an
ensemble might catch different anomalies than any single one does), or pushing the
alarm budget outside 5-50/month.

### All events (the paper's own metric)

The rest of this README, and most of `results.jsonl`, scores against every annotated
event: real anomalies plus 59x as many rare-but-nominal ones. Read these numbers as
"how well does this model track the mission's overall event stream", not "how well does
it catch anomalies".

Mission 2, full test period, pooled over ~20 monthly folds:

| Model | refit | online | hybrid | detections (online) |
|---|---|---|---|---|
| `GlobalSTD` | 0.752 | **0.969** | 0.965 | 329 in 20 months, precision 0.997 |
| `LaggedRidgeForecaster` | 0.678 | **0.969** | 0.968 | 327, precision 1.000 |
| `NearestNominalWindow` | 0.629 | 0.844 | **0.948** | 379 |
| `PCAReconstructor` | 0.669 | 0.899 | **0.910** | 326 |
| `Autoencoder-replay` | n/a | **0.945** | n/a | 315 |
| `SubsystemPCAEnsemble` | 0.593 | **0.835** | 0.832 | 393 |
| `Autoencoder` | 0.573 | 0.666 | **0.829** | 401 |
| `WindowedIsolationForest` | 0.228 | 0.322 | **0.331** | 686 |
| `LSTMForecaster` | 0.096 | 0.075 | **0.108** | 1229 |
| `HalfSpaceTrees` | 0.079 | 0.093 | **0.100** | 1826 |

**Paper-split** (L0): `GlobalSTD` / `LaggedRidgeForecaster` **0.984**, `Autoencoder`
0.983, `PCAReconstructor` 0.970, `NearestNominalWindow` 0.968, `SubsystemPCAEnsemble`
0.966 (all precision 1.000), vs `WindowedIsolationForest` 0.302, `HalfSpaceTrees`
0.087. Reference point: the paper's own best full-channel corrected event-wise F0.5 is
**0.071** (Telemanom-ESA-Pruned, all events); their cited 0.949 is an easier 11-channel
subset. Not fully apples-to-apples (L0 tunes on labeled data, which the paper avoids).

**Retraining cadence** (online, L0): monthly to bimonthly is the sweet spot. Weekly
folds calibrate thresholds on about 2 days of data, which is too noisy.

| Model | 1w | 2w | 1mo | 2mo |
|---|---|---|---|---|
| `GlobalSTD` | 0.830 | 0.888 | **0.969** | 0.909 |
| `PCAReconstructor` | 0.691 | 0.893 | 0.899 | **0.906** |
| `LaggedRidgeForecaster` | 0.630 | 0.928 | 0.969 | **0.973** |

`run_cadence.py` (weekly test-then-train, threshold recalibrated from a rolling
4-week buffer) sharpens this into updates vs. accuracy: `GlobalSTD` updating every 8
weeks scores 0.954 with 11 updates total, against 0.944 for 88 weekly updates. PCA's
drift-triggered strategy (a KS test on the score distribution) does best at 0.882 with
66 updates. Never updating collapses recall to 0.69/0.47. Updating matters; past
roughly monthly, extra updates buy nothing.

**Ablations** (`scripts/analyze_paper_stats.py`, from logged per-fold data):
- Online beats refit for every model tested, paired Wilcoxon p = 0.004-0.008.
- Event-wise calibration beats row-level calibration by +0.16 to +0.29 median F0.5 per
  fold, p ≤ 0.001.
- With event-wise calibration, alarm postprocessing is redundant: dropping it scores
  slightly higher (`GlobalSTD` 0.979 vs 0.969). It only rescues weaker calibration
  (row-level: 0.647 with vs 0.591 without) and stays load-bearing for the budget mode.
- Postprocess constants form a wide plateau: merge-gap 0-400 rows moves F0.5 by ≤0.021;
  min-duration is safe at 1-30 rows and only collapses at 100 (30 minutes).
- Seed variance is negligible: `Autoencoder-replay` over seeds {7, 42, 123} spans
  0.943-0.948.
- Cold-start isn't the L2 bottleneck: excluding the first 2 folds from scoring changes
  results by ≤0.006. The L2-to-L0 gap lives in the budget calibration itself.

**Channel-aware evaluation** (one threshold per channel, calibrated on a 6-fold
buffer):

| Model | global | channel-macro | subsystem-macro |
|---|---|---|---|
| `GlobalSTD` | 0.969 | 0.328 | 0.209 |
| `PCAReconstructor` | 0.899 | **0.374** | 0.230 |
| `LaggedRidgeForecaster` | 0.969 | 0.343 | 0.220 |

The drop is structural: a mission-level stream calibrates against ~300 events/month,
while a single channel sees about 6. OR-ing 47 per-channel streams floods precision
(0.04-0.10). Mission-level alerting is easy; channel attribution is the open problem,
the same wall the paper's own full-set results hit.

### Key findings

- Online training beats refit-from-scratch for every model.
- What you calibrate the threshold against matters as much as the model:
  `GlobalSTD-online` goes from 0.647 (row-level objective) to 0.969 (event-wise
  objective) with nothing else changed.
- The two best "all events" detectors are the two simplest: a per-channel k-sigma
  baseline and a closed-form linear lag-forecaster.
- None of that transfers to real anomalies specifically: see Anomalies only above.
  That's the actual open problem.
- `NearestNominalWindow` only works after per-channel standardization: raw window
  distance is dominated by large-scale channels (0.012 → 0.757 F0.5 from that one fix).
- A replay buffer fixes the online Autoencoder: plain `partial_fit` scores 0.666;
  rehearsing a 50k-row reservoir of past months at 20 epochs/fold scores 0.945.
- `channel_15` isn't pathological. It has 558 annotated events; the real issue is a
  train-nominal std of 8e-5 that widens 15x in the test period. Online stat updates
  absorb it; frozen-stats models don't.
- Telecommand features are neutral to mildly positive: `GlobalSTD`/`LaggedRidgeForecaster`
  unchanged, `PCAReconstructor` 0.899 → 0.935.

## Models

All semi-supervised: `fit`/`partial_fit` only ever see nominal data. All expose
row-level `score()`; most expose `score_per_channel()`/`most_anomalous_channels()`.

| Model | Idea |
|---|---|
| `GlobalSTD` / `OnlineGlobalSTD` | per-channel k-sigma, Welford-updated online |
| `PCAReconstructor` | linear reconstruction error, `IncrementalPCA` online |
| `SubsystemPCAEnsemble` | one PCA per subsystem, k-of-n combined |
| `LaggedRidgeForecaster` | ridge forecast of every channel from 3 lags of all channels; closed-form online solve |
| `NearestNominalWindow` | matrix-profile-style distance to nearest nominal window, bounded reservoir memory |
| `WindowedIsolationForest` | iForest over tumbling windows, grows trees online |
| `HalfSpaceTrees` | streaming mass-profile trees (Tan et al., IJCAI 2011) |
| `Autoencoder` (+ `-replay`) | nonlinear reconstruction; replay variant rehearses a reservoir of old months |
| `LSTMForecaster` | Telemanom-style forecasting error, downsized for CPU |
| `PyOD-ECOD` / `PyOD-COPOD` / `PyOD-IForest` | PyOD wrappers, unsupervised copula/isolation scores |
| `GDN` | graph-attention forecasting, learned channel graph (single-split only, `demo_gdn.py`) |

`scripts/run_cadence.py` compares retraining cadences prequentially: update every
1/4/8 weeks vs. drift-triggered vs. never.

## Mission 1 (prepared, not yet run)

Mission 1 (14 years at 30s resolution, 14.7M rows, 58 target channels) is the
out-of-domain validation of this whole methodology, staged in
`scripts/run_mission1.sh`:

| Stage | Time (8-core laptop) | What it does |
|---|---|---|
| `prep` | 1-2 h | ingest, preprocess, telecommand features |
| `core` | 7-9 h | walk-forward benchmark, paper-split, supervision ladder, rigor reruns |
| `extended` | 6-8 h | cadence sweeps, channel-aware, robustness grid, multi-seed |
| `analyze` | ~1 min | `analyze_paper_stats.py --mission 1` |

```bash
sh scripts/run_mission1.sh prep
nohup sh scripts/run_mission1.sh core > mission1_core.log 2>&1 &
nohup sh scripts/run_mission1.sh extended > mission1_ext.log 2>&1 &
sh scripts/run_mission1.sh analyze
```

Needs `scripts/download_raw_data.sh 1 && scripts/unzip_raw_data.sh` first (~12 GB,
already validated against the paper's reported Mission 1 statistics). All runs log to
the same `results.jsonl` with `mission: 1`; `analyze_paper_stats.py` filters by mission
so the two never mix. Mission 1 has long telemetry gaps; the engine skips folds with
too little data and reports how many it actually scored.

## Experiment tracking & reproduction

Every run appends to `data/experiments/results.jsonl`
(`esa_adb.experiments.log_experiment`; `--mlflow` mirrors to MLflow), including a
per-fold breakdown that `scripts/plot_fold_metrics.py` turns into a PNG under
`data/experiments/plots/` (gitignored, regenerate on demand).
`sh scripts/reproduce_experiments.sh 2` regenerates every table above. Seeds fixed at
42 throughout; `uv.lock` pins CPU-only PyTorch.


## Known approximations vs. the paper

- Thresholds are tuned on labeled validation data at L0/L1 (the detectors themselves
  never see labels). L2 doesn't do even that.
- Alarms are global, not channel-attributed; the paper's affiliation/ADTQC metrics
  aren't implemented.
- Telecommands are compact activity features (`--with-telecommands`), not the paper's
  per-TC binary channels.
- The paper's own "anomalies alone" numbers (Supplementary 4.2) aren't extracted here
  yet, so the anomalies-only table above has no published reference point to check
  against.

## License

[EUPL v. 1.2](LICENSE).
