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
online/streaming TSAD literature, and [`docs/leaderboard.md`](docs/leaderboard.md) for the
authoritative anomalies-only ranking with bootstrap confidence intervals (regenerate with
`scripts/analyze_leaderboard.py`). [`docs/pending_work.md`](docs/pending_work.md) lists what the
latest round finished and what is still open.

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

**A metric-convention note, found while reproducing these numbers this session**: the
leaderboard below reports `macro_f0.5_anomalies_only` (mean of per-fold F0.5, over only
the ~15 folds with >=1 real anomaly event); the detailed LOE/USDR/MemStream tables
further down report `pooled_event_wise_corrected_anomalies_only` (every fold's
predictions concatenated into one timeline, then scored once). Both are logged for
every run and both are legitimate, but they can disagree substantially for the SAME
model at the SAME budget -- e.g. `Autoencoder` (blind) is 0.057 macro / 0.084 pooled
here -- because a metric with this few positive events is sensitive to whether a
handful of high-false-alarm folds get one vote each (macro) or are weighted by their
actual (larger) share of total alarms (pooled). Not previously documented; check which
one a given number is before comparing across sections.

**Corrected this session**: a later full-catalog query of `results.jsonl` found that
`HeterogeneousFusion`, `RoCALite` and `OnlineIsolationForest` all beat `PyOD-ECOD` on
this metric — they just hadn't been pulled into this specific table before. See
`docs/leaderboard.md` for the authoritative, directly-queried ranking (also adds
`KalmanLocalTrend` 0.070 and `MatrixProfileDetector` 0.074, both new this session).
The table below is kept for its L0 column (not reproduced in the leaderboard) and is
otherwise superseded by `docs/leaderboard.md` for anomalies-only ranking purposes.

| Model | F0.5, all events (L0) | F0.5, real anomalies (best L2) | budget | recall | precision |
|---|---|---|---|---|---|
| `HeterogeneousFusion`\* | n/a | **0.125** | 50/mo | — | — |
| `RoCALite`\*\*\*\* | n/a | 0.114 | 10/mo | — | — |
| `OnlineIsolationForest` | — | 0.110 | 5/mo | — | — |
| `PyOD-ECOD` | 0.391 | 0.104 | 10/mo | 0.11 | 0.024 |
| `PyOD-IForest` | 0.314 | 0.098 | 50/mo | 0.19 | 0.049 |
| `LSTMForecaster` | 0.066 | 0.093 | 10/mo | 0.25 | 0.017 |
| `WindowedIsolationForest` | 0.339 | 0.092 | 5/mo | 0.10 | 0.049 |
| `PyOD-COPOD` | 0.260 | 0.091 | 10/mo | 0.11 | 0.049 |
| `SubsystemPCAEnsemble` | 0.894 | 0.083 | 10/mo | 0.27 | 0.041 |
| `CRTFAutoencoder`\*\*\*\* | n/a | 0.065 | 10/mo | 0.33 | 0.022 |
| `Autoencoder`\*\* | 0.669 | 0.057 | 5/mo | 0.11 | 0.079 |
| `GlobalSTD` | 0.897 | 0.056 | 20/mo | 0.23 | 0.041 |
| `USDR-SubsystemPCAEnsemble`\* | n/a | 0.048 | 20/mo | 0.32 | 0.009 |
| `NearestNominalWindow` | 0.765 | 0.047 | 20/mo | 0.42 | 0.014 |
| `LaggedRidgeForecaster` | 0.866 | 0.038 | 50/mo | 0.39 | 0.011 |
| `LOEAutoencoder-H`\*\*\* | n/a | 0.033 | 10/mo | 0.28 | 0.013 |
| `PCAReconstructor` | 0.858 | 0.023 | 20/mo | 0.32 | 0.007 |
| `LOEAutoencoder-S`\*\*\* | n/a | 0.020 | 10/mo | 0.33 | 0.008 |
| `MemStream`\*\*\* | n/a | 0.019 | 5/mo | 0.13 | 0.007 |
| `HalfSpaceTrees` | 0.069 | 0.012 | 50/mo | 0.13 | 0.016 |

**Updated again this session: the overall L2 leaderboard is now topped by
`TCopulaDetector` (0.160)**, a Student-t extension of `GaussianCopulaDetector`
(0.152, itself still the Mission-2 runner-up and the base this extension builds on) --
a novel-combination joint extension of `PyOD-ECOD`/`PyOD-COPOD`'s own empirical-CDF
framework (each channel's empirical-CDF value, probability-integral-transformed to a
quantile, then scored JOINTLY via a Mahalanobis-style distance against a fitted rank
correlation matrix, instead of summing independent per-dimension scores the way
ECOD/COPOD do -- ECOD's own authors flag that independence assumption as a limitation
and name copulas as their own suggested future work; the t-copula variant targets
Student-t marginals instead of Gaussian ones, for heavier-tailed sensitivity). Both
beat the `HeterogeneousFusion` ensemble below (0.125, previously the top of this
table) with a SINGLE model. The robustness fix this needed is
Spearman rank correlation plus the project's own MAD filter, the same
contamination-robustness problem the Kalman extension below already hit.

**Major caveat, also this session**: re-run on Mission 1 (a different spacecraft,
genuine out-of-domain check), BOTH copula variants collapse -- the opposite of the
generalization pattern `KalmanLocalTrend` and `PyOD-ECOD` show on the same check (both
score *higher* on Mission 1 than Mission 2). `GaussianCopulaDetector`'s anomalies-only
macro F0.5 drops more than 3x, from 0.152 (Mission 2) to 0.043 (Mission 1).
`TCopulaDetector` was then re-run on Mission 1 too, specifically to check whether its
heavier tails would hold up better: it drops nearly 3x, from 0.160 to 0.056 -- a real
improvement over the Gaussian variant's Mission-1 score (+30% relative) but nowhere
near enough to close the gap; both still land below `KalmanLocalTrend`/`PyOD-ECOD` on
Mission 1. **But this is copula-specific, not a blanket warning**: both
`KalmanBlockMultivariate` and `KalmanGlobalMultivariate` (this session's *other* two
joint-dependency extensions) were also re-run on Mission 1 and *improved* out of
domain instead (0.075 -> 0.093, +24%, and 0.073 -> 0.096, +31%), the same generalizing
direction `KalmanLocalTrend`/`PyOD-ECOD` show -- and `KalmanGlobalMultivariate` even
narrowly beats `KalmanBlockMultivariate` on Mission 1, reversing their Mission-2
order. `ADDAEILForest`, `HeterogeneousFusion` and `MatrixProfileDetector` were checked
too (0.089 -> 0.041, -54%; 0.125 -> 0.132, +5%; 0.074 -> 0.059, -20%): not a clean
two-way split after all once you look
beyond the Kalman family -- it's a spectrum, with Kalman generalizing most reliably,
`HeterogeneousFusion` generalizing weakly but positively, `MatrixProfileDetector`
degrading mildly, `Autoencoder`/`ADDAEILForest`
degrading moderately, and only the copula family collapsing outright. Ten models
now have a direct Mission 1
anomalies-only result. Treat
the copula rows of the Mission-2 ranking above as Mission-2-specific; the Kalman rows
have out-of-domain evidence behind them. Mission 1 results for these models are in the
Mission 1 section below.

**Two different "winners" live in this table -- don't conflate them.** Among the
models actually listed here, the leaderboard is topped by `HeterogeneousFusion`
(0.125, a per-fold rank-fusion ensemble of
`PyOD-ECOD`+`LSTMForecaster`+`WindowedIsolationForest`+`PyOD-COPOD`+`PyOD-IForest`),
not by any single model in THIS table -- see the note above for the true current best.
`CRTFAutoencoder` (0.065) only wins its own sub-category -- the
best of the four contamination-robust `Autoencoder` variants (itself,
`LOEAutoencoder-H`/`-S`, `USDR-SubsystemPCAEnsemble`, `MemStream`) -- it still sits well
below the top of the full table. "Best contamination-robust technique" and "best L2
model overall" are different questions with different answers here. See `docs/leaderboard.md`
for the full ranking, including new entries this session that also beat
`CRTFAutoencoder`'s own sub-category: `KalmanLocalTrend` (0.070, closed-form, no
training -- also the only model good at both this metric and all-events at once),
`MatrixProfileDetector` (0.074, training-free nearest-neighbor discord matching), and
`KalmanBlockMultivariate` (**0.075**, a block-Mahalanobis extension of
`KalmanLocalTrend` motivated by a dedicated Kalman-variant SOTA search: the independent
per-channel filter is structurally blind to an anomaly that breaks the NORMAL
correlation between two channels while each looks unremarkable alone; scoring each
subsystem's channels jointly via Mahalanobis distance against a robustly-estimated
block covariance closes that gap for a modest but real gain). A `KalmanGlobalMultivariate`
variant (same mechanism, one global 47-channel block instead of four subsystem blocks)
scores marginally lower (0.073) -- the "joint-across-everything beats per-subsystem"
pattern `PCAReconstructor` shows below does NOT transfer to this extension.
`DynamicFactorModel` (PCA loadings + a per-factor AR(1) Kalman filter) is a clean
negative result (0.054, below plain `PCAReconstructor`'s 0.089) despite its score only
ever increasing relative to plain PCA by construction -- factor dynamics shift the
budget-calibrated threshold to a worse operating point even though no individual row's
score went down.

\*Contaminated-data-robust variant (see [Contaminated training data](#contaminated-training-data-loe-usdr-memstream-and-crtf)
below); L0 wasn't run for it (the point is the label-free L2 setting), so its "all
events (L0)" column is left blank rather than filled with an apples-to-oranges number.
\*\*Corrected this session (was 0.052 @ 50/mo) -- see the reproducibility note directly
below and [Contaminated training data](#contaminated-training-data-loe-usdr-memstream-and-crtf).
\*\*\*Contaminated-data-robust variant, corrected/added this session for the same reason.
\*\*\*\*Contaminated-data-robust variant, added this session. Beats blind `Autoencoder`
on pooled anomalies-only F0.5 at every budget, confirmed across 2 seeds (42 and 123) --
see [Contaminated training data](#contaminated-training-data-loe-usdr-memstream-and-crtf)
for the full by-budget/by-seed picture, including a bigger finding: blind `Autoencoder`
itself swings >15x on this metric between seeds, so treat any single-seed number here
(this table included) as one draw, not a point estimate.

**A methodology caveat these corrected numbers exist because of**: `Autoencoder`,
`LOEAutoencoder-H/-S` and `MemStream` all train via PyTorch, and PyTorch's CPU intra-op
parallelism doesn't guarantee the same floating-point reduction order run to run.
For one forward/backward pass that's noise below any metric's resolution, but online
training here chains dozens of `partial_fit` calls across walk-forward folds, and
gradient descent is chaotically sensitive to exactly this kind of tiny perturbation:
confirmed empirically this session, the SAME seed=42 `Autoencoder` run produced
anomalies-only F0.5 values 2-4x apart depending only on CPU thread count / concurrent
machine load, with recall (which events get detected) matching almost exactly while
precision (how many total alarms got raised) drifted. `models/heavy/__init__.py` now
pins `torch.set_num_threads(1)` project-wide, which was verified to make an isolated
run and a run sharing the process with another PyTorch model produce bit-identical
results. `USDR-SubsystemPCAEnsemble` and every non-PyTorch model (`GlobalSTD`, PyOD
wrappers, `LaggedRidgeForecaster`, etc.) don't use PyTorch and were confirmed
bit-identical across repeated runs already, with or without the fix -- only the four
rows above needed correcting.

The ranking flips. `GlobalSTD` and `LaggedRidgeForecaster` top the all-events
leaderboard by nailing rare nominal events, then land near the bottom here. `PyOD-ECOD`,
near the bottom of the all-events table, comes out on top. Precision tops out at 7.9%:
of every 100 alarms these models raise, fewer than 8 point at a real anomaly. This is not a deployable detector (yet 🙂).

**How this compares to the published literature**: the ESA-ADB paper itself reports
"anomalies alone" numbers (Supplementary Material 4.2, Table 9) -- every algorithm the
paper's own authors tested, including their own all-events champion (`Windowed iForest`,
0.949 there), scores corrected event-wise F0.5 of **0.001 or less** on Mission2's full
channel set, real anomalies only. The paper's own text: *"no algorithm was able to
accurately identify [the] actual anomalies in this overabundance of rare nominal
events. This is the main challenge of this mission."* Every model in the table above
beats every number in their Table 9 by 1-2 orders of magnitude, under a harder protocol
(zero labels anywhere in L2, walk-forward over the full mission's 186 anomalies, vs.
their fixed split with a label-tuned threshold over just 9 anomalies). This isn't this
project underperforming a solved problem -- on this exact dataset, real-anomaly
detection is close to unsolved in the published literature, full stop. (A separate 2026
benchmark paper reports ECOD F0.5=0.448 on ESA-ADB, but on Mission1's easier 6-channel
"lightweight" subset and the all-events metric, not anomalies-only -- not a comparable
number to either this project's table or the original paper's own Table 9.)

**Ensemble/overlap, tried this session** (`scripts/run_l2_ensemble_overlap.py`): the top
5 anomalies-only models (each at its own best L2 budget above) detect mostly
*different* real anomaly events -- pairwise Jaccard over detected-event sets ranges
0.18-1.0 across the 10 model pairs, and the union of all 5 catches 12/186 events against
10/186 for the best single model. The gain is real but small and expensive: the OR of
all 5 alarm streams raises 3584 total detections (vs. a nominal combined budget of
85/month), so pooled anomalies-only F0.5 (0.018) is *worse* than several individual
models -- alarm-count cost outpaces the extra recall. Calibrating directly against
`anomalies_only`, and pushing the alarm budget outside 5-50/month, are both still
untried.

### Contaminated training data: LOE, USDR, MemStream, and CRTF

The models above all fit on a *label-filtered* nominal set even at "L2" -- filtering is
itself label information a real day-1 satellite doesn't have. Four techniques from the
contaminated-training-data / online-drift literature were integrated and tested, run
through the *exact* L2 protocol above (`--fit-on-all-rows --full-history --calibration
budget`, `online` variant, 15 scored folds, budgets 5/10/20/50/mo) so the numbers are
directly comparable to the table above.

**LOE ([Qiu et al., ICML 2022](https://arxiv.org/abs/2202.08088))** trains the
autoencoder itself to stop trusting its worst-reconstructed rows: each minibatch, the
`contamination`-fraction of rows with the largest gap between the normal and (bounded,
opposing) anomaly loss get routed to the anomaly loss instead of the normal one --
"hard" (`LOEAutoencoder-H`) makes a 0/1 call, "soft" (`LOEAutoencoder-S`) splits the
loss 50/50 to hedge. Full-history walk-forward, best-of-budget, **rerun this session
under `torch.set_num_threads(1)`** after the reproducibility issue described above was
found (the previously published table here had the verdict backwards -- see below):

| Model | F0.5, anomalies-only | budget | recall | F0.5, all events |
|---|---|---|---|---|
| `Autoencoder` (blind) | **0.084** | 5/mo | 0.11 | 0.640 |
| `LOEAutoencoder-H` | 0.020 | 5/mo | 0.20 | 0.623 |
| `LOEAutoencoder-S` | 0.009 | 10/mo | 0.33 | 0.340 |

**This reverses the previous verdict.** The original table (`Autoencoder` 0.052,
`LOEAutoencoder-H` **0.065**, "keep LOEAutoencoder-H") was generated before the CPU
thread-determinism issue above was found and fixed; under a bit-reproducible rerun
(confirmed identical whether `Autoencoder` trains alone or alongside another PyTorch
model in the same process -- previously it didn't), blind `Autoencoder` beats BOTH LOE
variants at every budget, on both anomalies-only and all-events. LOE's latent
reassignment doesn't help here once the comparison is actually apples-to-apples --
plausibly because the bounded-hinge anomaly loss substitute (below) is too weak a
signal on this backbone, but that's a hypothesis, not something this session verified.
**Verdict: don't reach for `LOEAutoencoder-H`/`-S` over plain `Autoencoder` for L2
deployment; the implementation stays (it's a faithful port of the paper's mechanism and
the caveat below is about the loss substitute, not a bug), but the case for using it
over the blind baseline was an artifact of unpinned CPU threading, not a real effect.**
Caveat unchanged: the paper never defines an anomaly loss for a plain reconstruction
autoencoder (only for MHRot/NTL/ICL self-supervised backbones), so
`loe_autoencoder.py`'s bounded-hinge `L_a = relu(margin - L_n)` is a reasonable but
not paper-validated substitute -- see `docs/literature_review.md` 5.5 for detail.

**USDR ([Ulmer, Zgraggen & Goren Huber, 2023](https://arxiv.org/abs/2308.13352))**
takes the opposite approach: instead of changing how a model trains, it refines the
*data* first. An ensemble of models trained on overlapping row subsets flags a row
suspicious when models that *excluded* it score it worse than models that *included*
it, then drops the worst offenders (MAD cutoff) before the real model ever fits. Wired
up as `USDR-SubsystemPCAEnsemble` (refines, then trains `SubsystemPCAEnsemble`):

| Model | F0.5, anomalies-only | budget | recall | F0.5, all events |
|---|---|---|---|---|
| `SubsystemPCAEnsemble` (unrefined) | **0.083** | 10/mo | 0.27 | 0.619 |
| `USDR-SubsystemPCAEnsemble` | 0.048 | 20/mo | 0.32 | **0.732** |

Refinement clearly helps the all-events metric (+18 to +25% at every budget) but *loses*
to the unrefined ensemble on anomalies-only at the two tighter, more realistic budgets
(5/mo: 0.020 vs 0.070; 10/mo: 0.045 vs 0.083) -- it only pulls ahead at 20-50/mo, where
the alarm budget is loose enough that recall stops being the bottleneck. Read together
with "Anomalies only: the number that matters" above, that's a real problem: refinement
is discarding some of the same hard-to-reconstruct rows that are the real anomalies, in
exchange for a cleaner fit against the mostly-benign event stream. (An initial
8-of-40-fold spot check during development looked better for USDR on both metrics --
worth remembering as a reminder that partial walk-forward runs are not a substitute for
scoring the full fold set before drawing a conclusion.) **Verdict: keep the
implementation (it's a faithful port and a real lever for the all-events metric), but
don't reach for it over plain `SubsystemPCAEnsemble` when anomaly recall at a tight
budget is the actual goal.**

**MemStream ([Bhatia et al., WWW 2022](https://arxiv.org/abs/2106.03837))** takes a
third approach: an encoder/decoder trains ONCE, then its weights freeze forever --
`partial_fit` does no gradient descent at all, only nearest-neighbor updates to a
bounded (2048-vector) memory of latent normal representations, and a streaming row only
ever gets written into memory if it *already* scores as normal-enough. That gating is
meant to buy contamination tolerance without LOE's `contamination` hyperparameter or
USDR's rejection step. Wired up as `models/heavy/memstream.py`, same protocol as above:

| Model | F0.5, anomalies-only | budget | recall | F0.5, all events |
|---|---|---|---|---|
| `Autoencoder` (blind) | **0.084** | 5/mo | 0.11 | 0.640 |
| `MemStream` | 0.019 | 5/mo | 0.13 | 0.434 |

Doesn't beat blind `Autoencoder` either, on the same paired run. Likely cause: `beta_`
(the "admit to memory" cutoff) is calibrated from the fit batch's OWN nearest-memory
distances -- at L2 that batch is already contaminated, so the gate ends up about as
permissive as the blind model's own reconstruction loss is forgiving, losing the
mechanism's theoretical edge. One genuine, confirmed advantage that doesn't show up in
this table: **no gradient step means `partial_fit` is markedly cheaper** -- observed
~3.7-4.8s/fold once past the first fold's pretraining, versus `Autoencoder`'s
~4.0-5.4s/fold doing real backprop every time, a real (if here overwhelmed) online-cost
lever from `docs/literature_review.md` 4's flagged MemStream reference. **Verdict: keep
the implementation as a cheap-online-update reference point, but it isn't earning a
deployment slot over blind `Autoencoder` on accuracy.** Caveat: this is NOT a verified
line-by-line port -- the paper's exact scoring/update equations weren't re-derived from
the source this session, only the architecture (frozen encoder + gated bounded memory)
described in `docs/literature_review.md` 4; see `memstream.py`'s own docstring.

**CRTF ([CLEANet](https://arxiv.org/abs/2510.22619), Zhang et al., 2025)** takes a
fourth angle, layered on top of the plain `Autoencoder` rather than a new backbone: each
epoch, cluster the CURRENT model's latent encodings (k-means, `n_clusters=8`), and give
every row a continuous weight from a sigmoid over its own reconstruction residual (no
fixed `contamination` rate, unlike LOE) -- high-residual rows count less toward both the
reconstruction loss AND a contrastive term that otherwise pulls each row toward its own
cluster's centroid. Wired up as `models/heavy/crtf_autoencoder.py`, same protocol:

**seed=42:**

| Budget | `Autoencoder` (pooled / macro) | `CRTFAutoencoder` (pooled / macro) | recall (blind / CRTF) |
|---|---|---|---|
| 5/mo | **0.084** / **0.057** | 0.023 / 0.024 | 0.113 / 0.199 |
| 10/mo | 0.010 / 0.014 | **0.027** / **0.065** | 0.333 / 0.333 |
| 20/mo | 0.007 / 0.014 | 0.007 / **0.019** | 0.376 / 0.339 |
| 50/mo | 0.006 / 0.019 | **0.007** / **0.022** | 0.376 / 0.376 |

**seed=123, rerun to check this wasn't a lucky seed:**

| Budget | `Autoencoder` (pooled / macro) | `CRTFAutoencoder` (pooled / macro) | recall (blind / CRTF) |
|---|---|---|---|
| 5/mo | 0.005 / 0.002 | **0.029** / **0.018** | 0.070 / 0.199 |
| 10/mo | 0.010 / 0.021 | **0.034** / **0.042** | 0.339 / 0.505 |
| 20/mo | 0.014 / 0.042 | **0.024** / **0.048** | 0.387 / 0.554 |
| 50/mo | 0.011 / 0.046 | **0.019** / 0.042 | 0.392 / 0.565 |

**Confirmed across both seeds: CRTF beats blind `Autoencoder` on pooled anomalies-only
F0.5 at every budget**, and on macro at 7 of 8 (budget/seed) combinations. This is the
first of the four contamination-robust techniques tried this session that survives a
second seed. There's a second, arguably more interesting finding hiding in these two
tables: **blind `Autoencoder`'s own best-budget pooled F0.5 swings 0.084 -> 0.005
(>15x) between seeds, while CRTF's swings only 0.027 -> 0.034 (~25%)** -- both runs are
individually bit-reproducible (the thread-determinism fix above holds), this is genuine
seed-to-seed variance in a metric computed from a handful of events. CRTF being
noticeably MORE stable across seeds than the model it's built on is itself indirect
evidence the contamination-resilience mechanism is doing something real, not just
getting lucky -- and it means any *single-seed* anomalies-only number for a
PyTorch-backed model in this project (including ones already in the main leaderboard
above) should be read as one draw from a wide distribution, not a point estimate.
**Verdict: promoted from "promising, unconfirmed" to the contamination-robust technique
worth reaching for over blind `Autoencoder` at 10-50/mo budgets** -- still only 2 seeds
and one dataset (Mission 2), so a paper-split cross-check and Mission 1 (once run)
remain open before calling this fully settled. Caveat: this is NOT a verified port
either -- only the paper's abstract was readable this session (full-text PDF extraction
failed), so "adaptive reconstruction weighting" and "clustering-guided contrastive
learning" are implemented here as a documented, good-faith interpretation of those two
names, not equations from the source; see
`crtf_autoencoder.py`'s own docstring.

**The other two papers reviewed this round didn't make it past research.**
"[An Encode-then-Decompose Approach...](https://arxiv.org/abs/2510.18998)" proposes a
genuine transformer-scale encoder (evaluated on an A800 GPU) -- too heavy for this
project's CPU-only, closed-form-first design budget, and worth reconsidering only if a
GPU-capable model tier is ever added. "[Applying ML to Routine Satellite Ground Segment
Operations...](https://elib.dlr.de/196102/)" (DLR's ATHMoS) is an operational system
description, not a new algorithm -- nothing concrete to port, though it does validate
this project's own bias toward simple, explainable detectors. Full writeups in
`docs/literature_review.md` 5.5.

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

All semi-supervised: `fit`/`partial_fit` only ever see nominal data (the `LOEAutoencoder`,
`USDR-SubsystemPCAEnsemble`, `MemStream` and `CRTFAutoencoder` exceptions are designed
to tolerate a contaminated fit set instead of assuming a clean one -- see [Contaminated
training data](#contaminated-training-data-loe-usdr-memstream-and-crtf)).
All expose row-level `score()`; most expose `score_per_channel()`/`most_anomalous_channels()`.

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
| `LOEAutoencoder-H` / `-S` | Autoencoder trained with [Latent Outlier Exposure](https://arxiv.org/abs/2202.08088): infers a latent anomaly label per row each minibatch instead of assuming the fit set is clean |
| `USDR-SubsystemPCAEnsemble` | `SubsystemPCAEnsemble` preceded by fold-local [USDR](https://arxiv.org/abs/2308.13352) refinement: an excluded-vs-included ensemble residual test drops the most suspicious training rows before fitting |
| `MemStream` | [Bhatia et al., WWW 2022](https://arxiv.org/abs/2106.03837)-inspired: encoder freezes after one pretrain, then a bounded latent memory updates online with no further gradient descent, gated to only ever admit already-normal-looking rows |
| `CRTFAutoencoder` | [CLEANet](https://arxiv.org/abs/2510.22619)-inspired: per-epoch k-means over the model's own latent space plus a continuous, residual-based sample weight (no fixed contamination rate) down-weight both the reconstruction loss and a cluster-centroid contrastive term |

`scripts/run_cadence.py` compares retraining cadences prequentially: update every
1/4/8 weeks vs. drift-triggered vs. never.

## Mission 1 (prepared; three-model out-of-domain check run this session)

Mission 1 (14 years at 30s resolution, 14.7M rows, 58 target channels) is the
out-of-domain validation of this whole methodology. The full staged ladder below is
still not run (compute cost), but this session ran the current L2 winners
(`KalmanLocalTrend`, `PyOD-ECOD`, blind `Autoencoder`) at L2 budget=20/mo over Mission
1's test period (2007-2014, 83 folds): the ranking direction from Mission 2 holds
(`PyOD-ECOD` 0.097 / `KalmanLocalTrend` 0.092 / `Autoencoder` 0.036 anomalies-only
macro F0.5), and both non-blind models score numerically *higher* on Mission 1's
anomalies-only metric than on Mission 2's -- the clearest out-of-domain evidence yet
that this project's conclusions aren't an artifact of Mission 2's specific channel set.
Two more top-5 models were checked later (budget 20/mo, anomalies-only macro F0.5):
`OnlineIsolationForest` 0.110 -> 0.047 and `RoCALite` 0.114 -> 0.020, both degrading strongly out of domain.
Staged full-ladder launcher:

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
- The paper's own "anomalies alone" numbers (Supplementary Material 4.2, Table 9) ARE
  now extracted -- see "Anomalies only" above for the comparison. Read with its own
  caveat: their test split has only 9 Mission2 anomalies (vs. this project's 186 across
  a full-mission walk-forward), so read their numbers as directional, not a tight bound.
- PyTorch-backed models' walk-forward numbers depend on CPU thread count (see the
  reproducibility note under "Anomalies only" above) -- pinned to 1 thread project-wide
  now, but any number generated before this session wasn't necessarily reproducible.

## License

[EUPL v. 1.2](LICENSE).
