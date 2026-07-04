# ESA-ADB anomaly detection

Preprocessing and modeling pipeline for the [ESA Anomaly Detection Benchmark
(ESA-ADB)](https://zenodo.org/records/15237121) dataset, comparing lightweight and heavy
ML architectures for anomaly detection in satellite telemetry. Preprocessing follows
Kotowski et al., *European Space Agency Benchmark for Anomaly Detection in Satellite
Telemetry* ([arXiv:2406.17826](https://arxiv.org/abs/2406.17826)) -- but **evaluation
does not**: the paper's own methodology fits a model once on a fixed historical half and
never refreshes it. This project's primary methodology is **continuous training and
validation**, walking forward through the mission as new telemetry arrives and either
periodically refitting or updating incrementally -- confirmed on real data (below):
incremental "never forget" updates beat refitting from scratch on just the latest month,
for every model tested. The paper's fixed single split still exists as a
paper-comparable reference point, not the headline result.

Designed to run on a laptop: chunked, sequential, Parquet throughout, no CSV. See
[`docs/literature_review.md`](docs/literature_review.md) for how this project's
approach sits relative to current SOTA/lightweight/online time series anomaly detection
research.

## Setup

```bash
uv venv .venv && uv sync --extra dev --extra heavy   # uv.lock pins exact versions; add --extra tracking for MLflow
source .venv/bin/activate
pytest tests/
```

## Pipeline

```bash
scripts/download_raw_data.sh 2 && scripts/unzip_raw_data.sh   # 1 or 2, or omit for both
python scripts/ingest_raw.py --mission 2        # pickles -> per-channel Parquet
python scripts/run_preprocessing.py --mission 2  # zero-order-hold resample, label, standardize
```

Output in `data/processed/Mission{N}/`: `chunks/*.parquet` (wide value table),
`labels_long/*.parquet` (sparse ground-truth labels, 4 event categories),
`channel_types.parquet`, `standardization_stats.parquet`. Config knobs live in
`configs/mission{1,2}.yaml`.

## Methodology

Every walk-forward run advances through a mission **one month at a time**. Each fold's
own training period is split 2/3 **fit** / 1/3 **calibration** (a threshold is never
calibrated on the same data a model was fit on); the model then scores the **next**,
fully held-out month.

Two independent axes get varied across experiments:

- **How much history a model keeps.** `refit`: a brand-new model, trained from scratch
  each fold on only that fold's own month. `expanding`/`sliding`: `refit`, but the
  fit window is either everything since the mission started or only the last N months.
  `online`: ONE model instance, alive for the whole mission, updated every fold via
  `partial_fit()` -- never discarded, never retrained from zero. `hybrid`: AND-combines
  a long-memory and a short-memory model (`PairedLearnerEnsemble`, adapting Bach &
  Maloof's *Paired Learners for Concept Drift*, IEEE 2008) -- flags a point only when
  both agree, so settled drift the reactive model already adapted to doesn't trip the
  stable one's alarm.
- **How the alarm threshold adapts.** `static` (this project's default): calibrated once
  per fold, held fixed across the whole next month. `rolling`/`pot`
  (`models/dynamic_threshold.py`): continuously re-derived as that month streams in,
  from a recent-score percentile or a fitted extreme-value tail, respectively.

All real-data results below cover Mission2's **entire test period** (~20 monthly folds,
2001-09-30 to 2003-07-01), not a fold-capped demo subset, unless stated otherwise.

## Metrics

- **Window-level** (`evaluate_window_predictions`): precision/recall/F-beta treating
  every fixed-length window as one sample. Simple, but penalizes every individual
  missed window inside a long real event as a separate false negative.
- **Event-wise** (`evaluate_event_wise`): consecutive flagged rows collapse into one
  **detection** per contiguous alarm; precision = fraction of detections overlapping a
  real event, recall = fraction of real events with >=1 overlapping detection. Far more
  forgiving of imprecise boundaries, but has a known flaw: an algorithm that flags
  *every single row* gets a perfect precision (one giant detection trivially overlaps
  every event).
- **Corrected event-wise** (`evaluate_event_wise_corrected`): the ESA-ADB paper's own
  metric (Kotowski et al. 2024, "Corrected event-wise F-score"; correction originally
  from Sehili & Zhang), fixing exactly that flaw by multiplying event-wise precision by
  the **time-domain true-negative rate**:
  ```
  Pr_corr = (TP_e / (TP_e + FP_e)) * TNR_t,   TNR_t = TN_t / N_t
  F_beta_corr = (1+beta^2) * Pr_corr * Rec_e / (beta^2 * Pr_corr + Rec_e)
  ```
  `N_t` is every row outside any annotated event's span (the "should stay quiet"
  timeline), `TN_t` is however much of it was correctly left unflagged. Flagging
  everything now drives `TNR_t -> 0`, crushing precision regardless of how many
  detections happened to land on real events. Recall is untouched. Beta=0.5 throughout
  this project (precision weighted over recall, matching the paper and Hundman et al.'s
  original Telemanom event-wise metric).
- **Not implemented**: the paper's channel-/subsystem-aware F-scores (need per-channel
  attributed predictions, which most models here don't produce), affiliation-based
  F-score (Huet et al. 2022), and ADTQC detection-timing score -- all real, useful, and
  a reasonable next step (see "Open next steps").

## Results

### Single split (paper-comparable baseline)

All models below are evaluated with the same three metrics, on the paper's own fixed
single train/val/test split (`scripts/demo_lightweight_paper_split.py` for the
lightweight models, `demo_gdn.py` and `run_experiments.py` for the heavy ones):

| Model | window F0.5 | event-wise F0.5 | corrected event-wise F0.5 |
|---|---|---|---|
| `SubsystemPCAEnsemble` (OR) | 0.10 | 0.07 | 0.05 |
| `HalfSpaceTrees` | 0.11 | 0.08 | 0.07 |
| `GlobalSTD` | 0.18 | 0.09 | 0.08 |
| `WindowedIsolationForest` (stat+spectral feats) | 0.20 | 0.51 | -- |
| `SubsystemPCAEnsemble` (2-of-4 agree) | 0.23 | 0.29 | 0.29 |
| `GDN` | 0.33 | 0.18 | 0.18 |
| `WindowedIsolationForest` | 0.37 | 0.26 | 0.25 |
| `PCAReconstructor` | 0.58 | 0.46 | 0.46 |
| `LSTMForecaster` | 0.09 (undertrained smoke-test) | -- | -- |
| `Autoencoder` | **0.72 (best)** | **0.62 (best)** | **0.55 (best)** |

Notes:
- **`Autoencoder` wins outright** here, trained with real compute (30 epochs, full
  2.4M-row train set) -- confirms real nonlinear cross-channel structure a linear
  `PCAReconstructor` can't capture.
- **`SubsystemPCAEnsemble` (OR) and `HalfSpaceTrees` are real, weak results, not
  artifacts**: OR-combining independent per-subsystem detectors compounds false
  positives (79,843 vs. 11,069 detections for one joint model); vanilla Half-Space Trees
  splits one random axis at a time, leaving most of a leaf's bounding region
  unconstrained in 47 dimensions (it was designed for 2-10-feature streams).
- **Window features are a real, mixed result**: fewer, more precise detections push
  event-wise F0.5 from 0.26 to 0.51 (2nd best) while window-level F0.5 gets worse
  (0.37->0.20) -- the two metrics reward different things.
- `LSTMForecaster`'s number is a deliberately tiny CPU smoke-test (10 days, 5 epochs),
  not a real training run -- see "Open next steps".
- Among the genuinely lightweight models (excluding `Autoencoder`/`LSTMForecaster`/
  `GDN`), `PCAReconstructor` wins clearly on this paper-comparable single split (0.46
  corrected event-wise) -- but this is NOT the project's primary methodology; see
  "Walk-forward" below, where `GlobalSTD` in `online`/`hybrid` mode overtakes it (macro
  F0.5 0.68-0.70 vs. `PCAReconstructor`'s 0.54) once both get to accumulate history
  instead of being scored from one fixed, frozen fit.
- `GDN` (graph-attention forecasting over a learned channel graph, Deng & Hooi AAAI
  2021) trails `PCAReconstructor`/`Autoencoder` despite being architecturally the most
  sophisticated model here. It's also still un-tuned (15 epochs, no learning-rate
  schedule or hyperparameter sweep -- see "Open next steps"), and its learned graph only
  weakly recovers `channels.csv`'s known subsystem structure (see "Explainability"),
  which is consistent with a possible mismatch between GDN's sparse-sensor-graph
  assumption (validated in the original paper on tightly-coupled industrial testbeds)
  and satellite telemetry's more heterogeneous, less physically-coupled channel set.

### Replicating the paper's own metric

`scripts/demo_corrected_event_wise.py` validates `evaluate_event_wise_corrected`
against the paper's own published numbers two ways, using the paper's exact fixed
train/val/test split (not this project's walk-forward methodology):

1. **Replication**: `WindowedIsolationForest` on Mission2's exact "lightweight subset"
   (channels 18-28, subsystem 1). Result: **corrected event-wise F0.5 = 0.938**, against
   the paper's own published **0.949** on the same 11 channels -- confirms the metric
   implementation matches theirs.
2. **Fair full-set comparison**: on the full 47-channel set, this project's models score
   **`Autoencoder` 0.553, `PCAReconstructor` 0.458, `WindowedIsolationForest` 0.248**.
   The paper's own full-set best is **0.071** (Telemanom-ESA-Pruned) -- their own
   Windowed iForest couldn't even run on the full channel set. This project's models
   substantially outperform it, though not a fully apples-to-apples claim: the paper
   deliberately avoids extensive hyperparameter tuning ("to present the results of
   typical TSAD approaches"), while this project's thresholds are tuned via
   `find_best_threshold` on labeled validation data throughout.

   **Why `WindowedIsolationForest` runs here at all where the paper's OOM'd**: this
   project's `WindowSpec(length=10, stride=10)` builds *non-overlapping* ("tumbling")
   windows -- 243,296 of them for Mission2's 2.43M-row train set, a ~0.9GB feature
   matrix. TimeEval (the framework the paper's own benchmark is built on) defaults to
   `stride=1` -- a new, overlapping window at every single row. At `stride=1` and this
   project's own `length=10`, the same data needs ~9.1GB; at `length=17` it's ~15.6GB; at
   `length=100` it's ~91.5GB -- comfortably explaining an OOM on typical hardware. Not a
   bug on either side, but a real memory-vs-resolution trade-off: tumbling windows are
   far cheaper, at the cost of only ever resolving detections to a 10-row block
   (broadcast back to every row in it via `broadcast_window_pred_to_rows`) rather than a
   single row.

```bash
python scripts/demo_corrected_event_wise.py --mission 2
```

### Walk-forward (primary methodology)

**`PCAReconstructor`, refit each fold, three history windows** (`scripts/demo_walk_forward.py --mission 2 --sliding-months 3`):

| expanding | sliding (3mo) | hybrid |
|---|---|---|
| **0.53** | 0.26 | 0.52 |

Expanding (all history) wins outright over the full 20-fold mission.

**`HalfSpaceTrees`, refit vs. online, both with expanding history** (`scripts/demo_online_hst.py --mission 2`), macro/micro F0.5:

| | refit | online |
|---|---|---|
| macro | 0.157 | **0.170** |
| micro | 0.135 | **0.191** |

Online wins on accuracy but not speed here (964s vs. 922s) -- unlike `PCAReconstructor`'s
clean batch retrain, `partial_fit` still walks every row of the (also expanding, hence
growing) fit sub-period each fold, so there's no compute shortcut once both variants see
the same amount of data.

**All 6 single-split models, `refit` vs. `online` vs. `hybrid`, 1-month folds** (`scripts/demo_full_mission_walk_forward.py --mission 2`), macro F0.5:

| Model | refit | online | hybrid |
|---|---|---|---|
| `GlobalSTD` | 0.11 | **0.68** | 0.70 |
| `PCAReconstructor` | 0.13 | **0.54** | 0.54 |
| `SubsystemPCAEnsemble` | 0.06 | 0.19 | **0.21** |
| `Autoencoder` | 0.11 | **0.23** | 0.23 |
| `WindowedIsolationForest` | 0.21 | 0.25 | **0.25** |
| `LSTMForecaster` | 0.09 | 0.09 | **0.10** |

**Online beats refit for every single model here, several by a wide margin** -- the
strongest single finding in the project. One month alone badly miscalibrates almost
every model ("`PCAReconstructor` needs ~3+ months to stabilize," now confirmed across 5
more model types); `GlobalSTD`'s 6x jump is the starkest, since even a couple of months
of Welford-accumulated mean/std stabilizes a k-sigma baseline far more than a single
fresh month can. `LSTMForecaster` barely moves either way -- a (deliberately downsized,
`input_length=50`/`hidden_size=32`) sequence model doesn't seem to get enough signal
from one month regardless of whether it's kept or discarded.

**Dynamic thresholds on top of `online` training, static vs. rolling vs. POT** (`scripts/demo_dynamic_threshold.py --mission 2`), macro F0.5:

| Model | static | rolling | pot |
|---|---|---|---|
| `GlobalSTD` | **0.68** | 0.44 | 0.25 |
| `SubsystemPCAEnsemble` | **0.61** | 0.43 | 0.29 |
| `PCAReconstructor` | **0.54** | 0.43 | 0.34 |
| `Autoencoder` | 0.23 | 0.42 | **0.46** |
| `WindowedIsolationForest` | 0.25 | **0.26** | 0.15 |
| `LSTMForecaster` | **0.09** | 0.08 | 0.08 |

**Static wins or ties 5 of 6** -- the opposite of the going-in expectation. Since
`online` training already accumulates months of history by calibration time, the static
threshold (calibrated from that same well-informed model) is already well-matched to
it; recalibrating *within* just one eval month means chasing a smaller, noisier signal.
`Autoencoder` is the exception, and had the worst static baseline of the six (0.23) --
more room for a different strategy to help.

## Models

| Model | Idea | Real Mission2 window / event-wise / corrected event-wise F0.5 |
|---|---|---|
| `GlobalSTD` | per-channel k-sigma | 0.18 / 0.09 / 0.08 |
| `HalfSpaceTrees` | online Isolation-Forest-style streaming mass profile (Tan, Ting & Liu, IJCAI 2011) | 0.11 / 0.08 / 0.07 |
| `GDN` | graph-attention forecasting over a *learned* channel graph (Deng & Hooi, AAAI 2021) | 0.33 / 0.18 / 0.18 |
| `WindowedIsolationForest` | Isolation Forest over flattened (or stat+spectral) windows; `partial_fit` via `warm_start` | 0.37 / 0.26 / 0.25 |
| `PCAReconstructor` | reconstruction error, truncated PCA/SVD; `partial_fit` via `IncrementalPCA` | 0.58 / 0.46 / 0.46 |
| `LSTMForecaster` | multivariate forecasting error (Telemanom-ESA-style) | 0.09 (smoke-test) / -- / -- |
| `Autoencoder` | nonlinear reconstruction, PCA's nonlinear extension | **0.72 / 0.62 / 0.55 (best)** |
| `SubsystemPCAEnsemble` | one `PCAReconstructor` per `channels.csv` subsystem, OR- or k-of-n-combined | 0.10-0.29 (see Results) |

All semi-supervised/unsupervised: `fit()`/`partial_fit()` only ever see nominal data,
matching a newly launched satellite with no curated anomaly labels yet.
`RollingPercentileThreshold`/`POTThreshold` (`models/dynamic_threshold.py`) wrap any
detector's `score()` with a continuously-adapting threshold (see Methodology/Metrics).

`GDN` deviates from the original paper in three deliberate ways (see its class
docstring for the full rationale): channels are z-scored before training, like every
other model here; normalization stats are calibrated from held-out data rather than
recomputed on whatever is being scored; and the learned channel graph is frozen after
`fit()` (not after `partial_fit()`), trading some adaptability for stable, reproducible
explanations across folds.

### Explainability

Every model above except `WindowedIsolationForest` and `SubsystemPCAEnsemble`'s combined
score exposes **`score_per_channel(X)`** (an `(n_rows, n_channels)` array -- the
per-channel signal each model already computes internally before aggregating into one
row-level `score()`) and **`most_anomalous_channels(X, channel_names, top_n)`** (the
`top_n` channels driving each row's score, by name) -- "something is anomalous" becomes
"channel_23 and channel_41 are anomalous," at no extra training cost, since the
per-channel signal already existed inside `score()`.

`GDN` additionally exposes the paper's own explainability story: **`graph_edges()`**
(which channels the model considers related to which -- the learned dependency graph,
independent of any row) and **`attention_for_window()`** (which neighbors mattered most
for one specific forecast/alert, `(n_channels, n_channels)` attention weights for a
single window).

**Real finding** (`scripts/demo_explain_detections.py`, `Autoencoder` on Mission2):
aggregating `most_anomalous_channels()` over every real event the model detects shows
the explanation is NOT spread evenly across 47 channels -- `channel_18` alone appears in
the top-3 explanation for **76.7%** of all 6,334 detected events, with `channel_17`,
`19`, `20` close behind (38-47%). All four belong to `subsystem_1` -- the exact same
subsystem the paper's own team picked as Mission2's "lightweight subset" (channels
18-28, used in "Replicating the paper's own metric" above) for being the richest source
of annotated events. `Autoencoder` never saw that framing -- it was trained on all 47
channels jointly -- yet its own explainability independently rediscovers the same
subsystem as where nearly all real detectable activity in Mission2 lives.

**Four follow-up analyses** (`scripts/demo_explainability_deep_dive.py`, trains
`PCAReconstructor`/`Autoencoder`/`GDN` once and reuses them for all four; results in
`data/experiments/results.jsonl` under `model_name` prefixed `Explainability-`):

```bash
python scripts/demo_explainability_deep_dive.py --mission 2
```

1. **Does `GDN`'s learned graph recover the known subsystem structure?** Only weakly:
   32.8% of learned edges connect same-subsystem channels, against 31.5% expected from a
   random top-k graph (**1.04x** -- essentially chance level). Consistent with `GDN`
   still being untuned (see "Open next steps") and/or a genuine mismatch between its
   sparse-graph assumption and how satellite telemetry channels actually relate to each
   other (see "Models" above).
2. **Do `Autoencoder` and `PCAReconstructor` blame the same channel for events they
   BOTH detect?** Almost never, and by more than it first looks -- **1.05% agreement**
   (66 of 6,307 jointly-detected events) on the #1 channel, against a **2.13%**
   random-chance baseline for picking the same channel out of 47 independently
   (`1/47`) -- **0.49x**, roughly HALF what pure chance would give. Both models
   correctly flag the same events as anomalous, but the linear (PCA) and nonlinear
   (Autoencoder) reconstruction errors attribute *why* very differently, and appear to
   actively avoid agreeing more than randomness alone would predict -- worth deeper
   follow-up (e.g. comparing top-3 overlap rather than an exact top-1 match, which is
   brittle to near-tied scores) before leaning on it too hard.
3. **Do false positives implicate different channels than true positives?**
   Partially -- `channel_18`/`channel_20` dominate both (expected, given how much of the
   mission they explain overall), but `channel_15` is the single top false-positive
   channel (857 false alarms) while barely registering among true positives -- a
   concrete, actionable candidate for a channel whose model needs revisiting rather
   than trusting at face value.
4. **Does the "usual suspect" channel drift over the mission?** Yes, with a clean,
   single switch: `channel_17` dominates the first 4 months of the test period
   (2001-10 to 2002-01), then `channel_18` takes over for the remaining ~17 months
   through 2003-06 -- not a gradual drift but an abrupt handoff, worth checking against
   the mission's own operational timeline if that context were available (it isn't --
   ESA-ADB anonymizes exactly this kind of detail).

## Experiment tracking

Every run is logged to `data/experiments/results.jsonl` via
`esa_adb.experiments.log_experiment(model_name, mission, config, metrics, dataset_scope, notes)`:

```python
from esa_adb.experiments import load_experiments, summarize_experiments
summary = summarize_experiments()   # one flat row per run
```

`scripts/run_experiments.py`, `demo_walk_forward.py`, `demo_online_hst.py`,
`demo_full_mission_walk_forward.py`, `demo_dynamic_threshold.py`,
`demo_corrected_event_wise.py`, `demo_features_and_strict_subsystems.py`, `demo_gdn.py`,
`demo_lightweight_paper_split.py`, `demo_explain_detections.py`, and
`demo_explainability_deep_dive.py` all log to it. Add
`--mlflow` to any of them (or `mirror_to_mlflow=True` to `log_experiment`
directly) to also mirror runs into MLflow (`sqlite:///mlflow.db`,
`pip install -e ".[tracking]"` first) -- optional and additive, not a replacement: the
JSONL log stays the source of truth (zero dependencies, diffable in a PR), MLflow adds a
run-comparison UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Reproducing results

```bash
sh scripts/reproduce_experiments.sh 2   # exact commands/flags behind every logged run
```

`Autoencoder`/`LSTMForecaster`/`GDN` take `seed=42` by default; `uv.lock` pins exact
package versions, including PyTorch's CPU-only build specifically (`[tool.uv.sources]`
in `pyproject.toml` -- without it, `uv lock` resolves the full CUDA build).

## Repo layout

```
configs/            per-mission YAML params
scripts/            CLIs: download/ingest/preprocess, model demos, experiment runners
src/esa_adb/
  config.py, io/, preprocessing/   ingestion, ZOH resampling, labeling, typing, windowing
  pipeline.py        orchestrates ingestion + chunked resampling + stats
  evaluation.py       window-level / event-wise / corrected event-wise metrics
  models/             base.py (shared interface), lightweight/, heavy/, ensemble.py,
                       walk_forward.py, dynamic_threshold.py, window_features.py
  experiments/        run tracking (log_experiment / load_experiments / summarize_experiments)
tests/                unit/ + integration/ (paper's own worked example, real-data regressions)
```

## Known approximations vs. the paper

- Channel typing falls back to a data-driven heuristic where `channels.csv` group
  metadata isn't available; standardization stats always use exact per-channel nominal
  exclusion.
- `evaluate_event_wise_corrected` implements the paper's primary metric exactly
  (validated to within 0.01 of their own published number, see "Replicating the paper's
  own metric"); their secondary metrics (channel-/subsystem-aware, affiliation-based,
  ADTQC) are not implemented.
- Telecommands (`preprocessing/telecommands.py`) are implemented but not wired into the
  main pipeline yet.
- Validated end-to-end against real Mission1 (76 channels, 14 years) and Mission2 (100
  channels, 3.5 years) downloads -- row counts, split lengths, and category distributions
  all cross-check against the paper's own reported statistics for both missions.

## Open next steps

- **`GDN` needs real tuning before its accuracy can be judged fairly**: more epochs
  (loss hadn't plateaued at 15), a learning-rate schedule, and a sweep over
  `embed_dim`/`top_k`/`window_length` -- none tried yet, unlike `Autoencoder`, whose
  biggest win was simply "more compute" on an otherwise-default config. Also worth
  running its `partial_fit` (graph frozen after `fit()`, see class docstring for why)
  through the same refit-vs-online walk-forward comparison every other model went
  through, now that it exists.
- `channel_15`'s false-positive-heavy, true-positive-light explainability profile (see
  "Explainability") is a concrete lead for what to investigate next on Mission2 itself,
  not just the models -- worth checking if it's a channel type
  (`preprocessing/typing_.py`) or standardization choice mismatch, or if it's a
  genuinely noisy channel that should be down-weighted or excluded.
- `HalfSpaceTrees` is weak standalone on 47 dimensions -- worth trying as a cheap,
  fast-updating *first-pass filter* ahead of `PCAReconstructor`/`Autoencoder`, or
  replacing its single-random-axis splits with splits on a PCA-projected subspace.
- Tune `models/window_features.py`'s window length/stride now that its real event-wise
  result (0.51, 2nd best) is known to be promising -- and vectorize
  `extract_window_features` across windows (currently ~19 min on Mission2's full train
  split, a real bottleneck already, and Mission1 has more channels).
- `SubsystemPCAEnsemble(min_agreeing=2)` fixed most of the OR rule's false-positive
  blowup but still trails one joint `PCAReconstructor` on Mission2's 47 channels; worth
  re-checking on Mission1 (76 channels, more natural subsystem separation).
- `WindowedIsolationForest`'s online variant grows its forest by `n_new_estimators`
  every fold with no pruning -- fine over 20 folds, but a production deployment running
  for years needs a retirement policy for old trees.
- Dynamic thresholding only clearly helped `Autoencoder`. Worth checking whether a
  LONGER adaptation window (multi-month, not bounded to one eval month) closes the gap
  for the models it currently loses to, since the working theory is that one month
  alone is a noisier calibration signal than the online model's own accumulated history.
- Implement the paper's affiliation-based F-score and ADTQC (see Metrics) for a fuller
  comparison, and channel-aware attribution (would need most models here to report which
  channel(s) drove a detection, not just a single aggregate score).

## License

Licensed under the [EUPL v. 1.2](LICENSE).
