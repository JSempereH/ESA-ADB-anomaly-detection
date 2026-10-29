# Literature review: unsupervised, online anomaly detection in time series

Scope: recent (mostly 2023-2026) work on **unsupervised/semi-supervised** multivariate
time series anomaly detection, prioritizing methods that are (or could become) **online**
(train and/or infer as data streams in, not a single fixed batch fit) -- the methodology
this project uses instead of the ESA-ADB paper's own fixed-split benchmark (see main
README's "Methodology"). Extra weight given to satellite/spacecraft telemetry work and to
lightweight/edge-deployable models, both directly relevant to this project's own
constraints (laptop-friendly, "day 1 of a newly launched satellite" framing).

## 1. The central finding: true online training is a real, acknowledged gap

The most directly relevant source found is Correia et al., *Online Model-based Anomaly
Detection in Multivariate Time Series: Taxonomy, Survey, Research Challenges and Future
Directions* ([arXiv:2408.03747](https://arxiv.org/abs/2408.03747), 2024, Mercedes-Benz AG
/ Leiden University). It proposes exactly the taxonomy this project's own README now
uses -- a clean split between:

- **online training** (OT): the model's parameters keep updating as new data streams in
  (this project's `partial_fit`/"online" walk-forward variant);
- **online inference** (OI): the model scores each new point as it arrives, without
  waiting for a full sequence (this project's chunked `score()` calls);
- **detection without labels** (DwL): unsupervised/semi-supervised, no anomaly labels
  needed to fit (every model in this project).

Their own survey table (Table 5/6) shows almost no published method satisfies all three
at once. Most tellingly, **Section 6.1 states explicitly: "Approaches based on
autoencoders that are trained and infer in an online fashion do not exist, to the best
of the author's knowledge"** -- i.e. as of a 2024 survey, nobody had published an
autoencoder-based detector with genuine online training AND online inference. This
project's `Autoencoder.partial_fit()` + walk-forward "online" evaluation is, per this
survey, in territory the literature had explicitly flagged as unaddressed. Worth citing
as the gap this project's methodology fills.

## 2. SOTA unsupervised architectures (mostly offline-trained, for context)

These are the reference points most current work benchmarks against -- none are online
in the OT sense above, but they define "what accuracy looks like" on this problem.

| Model | Idea | Notes |
|---|---|---|
| **OmniAnomaly** (Su et al., KDD 2019) | GRU + VAE, stochastic latent variable model, reconstruction probability as score | Best on noisy datasets (e.g. WADI) per its own stochasticity modelling; still a common baseline. |
| **USAD** (Audibert et al., KDD 2020) | Two autoencoders in an adversarial (GAN-like) training scheme | Fast to train; a natural point of comparison for this project's plain `Autoencoder`. |
| **TranAD** ([arXiv:2201.07284](https://arxiv.org/abs/2201.07284), Tuli et al., VLDB 2022) | Transformer encoder-decoder + self-conditioning + adversarial training to amplify reconstruction error | Addresses USAD's local-context-window limitation via full-sequence self-attention; widely used as a current strong baseline. |
| **Anomaly Transformer** ([arXiv:2110.02642](https://arxiv.org/abs/2110.02642), Xu et al., ICLR 2022) | "Association discrepancy": a learnable Gaussian prior-association vs. self-attention series-association, exploiting that real anomalies can't build non-trivial long-range associations | SOTA on several benchmarks including a "space & earth exploration" category; a 2025 follow-up, **MAAT** ([arXiv:2502.07858](https://arxiv.org/html/2502.07858v1)), swaps in a Mamba backbone for better long-sequence efficiency. |
| **GDN** (Deng & Hooi, AAAI 2021) | Learned sensor graph + graph attention forecasting (this project's `models/heavy/gdn.py`) | Its own paper lists "online training methods" as unaddressed future work -- confirmed no follow-up has filled that gap (see  4). |

## 3. Emerging direction: time series foundation models (2025-2026)

A genuinely new trend since ESA-ADB's own 2024 paper: pretrained, zero-shot-capable
models applied to anomaly detection without per-dataset training.

- **MOMENT**, **TimesFM** (Google), **TS-Pulse** (IBM, TSMixer-based) -- general-purpose
  pretrained time series backbones, anomaly detection as one of several downstream tasks.
- **TimeRCD** ([arXiv:2509.21190](https://arxiv.org/abs/2509.21190), 2025) -- zero-shot
  anomaly detection via synthetic-data pretraining and "relative context discrepancy."
- **TimeRadar** ([arXiv:2602.19068](https://arxiv.org/pdf/2602.19068), KDD 2026) --
  "domain-rotatable" foundation model, explicitly targeting cross-domain generalization
  (e.g. train on server metrics, deploy on sensor/telemetry data without fine-tuning).
- **VETime** ([arXiv:2602.16681](https://arxiv.org/pdf/2602.16681), 2026) --
  vision-enhanced zero-shot detection.

Relevance to this project: zero-shot generalization is the opposite trade-off from this
project's "day 1 of a newly launched satellite" framing (which assumes *some* nominal
data exists to fit on, just no curated anomaly labels) -- but a foundation model could
plausibly replace `fit()` entirely for a satellite with even less initial data than this
project assumes. Not tried here; a real, if heavy, next step (these are pretrained on
generic corpora, not validated on satellite telemetry specifically as far as found).

## 4. Truly online (train + infer online) methods

Where this project's own approach has the most real company:

- **Half-Space Trees** (Tan, Ting & Liu, IJCAI 2011) -- this project's own
  `HalfSpaceTrees`. Implemented in the [`river`](https://riverml.xyz/) streaming-ML
  library as `HalfSpaceTrees`/`QF_HST`/`TF_HST` variants; `river` is the closest thing to
  a standard toolkit for this project's methodology and could replace some hand-rolled
  code here if its API constraints (mainly: designed for single-sample-at-a-time
  streaming, not this project's chunked-batch convention) turn out to be worth the fit.
- **Robust Random Cut Forest** (Guha et al., ICML 2016) -- the other classic streaming
  Isolation-Forest relative, maintains a dynamically-updated sketch of the stream rather
  than Half-Space Trees' fixed-structure mass profile. Not implemented in this project;
  a reasonable second "online, tree-based, lightweight" candidate alongside
  `HalfSpaceTrees` for the "Open next steps" comparison already planned.
- **Online Isolation Forest** ([arXiv:2505.09593](https://arxiv.org/html/2505.09593v1),
  2025) -- a genuinely online-native Isolation Forest (splits and rebalances trees
  incrementally), distinct from the older "iForestASD" pattern (periodic full batch
  refit, discard old ensemble -- the exact limitation this project's own
  `WindowedIsolationForest.partial_fit()` via `warm_start` was designed around, see
  README Models). Worth a direct look: might be a more principled `partial_fit` than the
  `warm_start`-based growing-forest approach currently used here.
- **MemStream** ([arXiv:2106.03837](https://arxiv.org/abs/2106.03837), Bhatia et al.,
  WWW 2022) -- a memory-augmented autoencoder for streaming anomaly detection with
  explicit concept-drift handling via a bounded memory module. The closest published
  thing to "the autoencoder that doesn't exist yet" flagged in  1. **Integrated** as
  `models/heavy/memstream.py` (runner `MemStream`) -- see 5.5 for the L2 result
  (doesn't beat blind `Autoencoder` on this backbone/dataset, but its no-backprop
  `partial_fit` is confirmed markedly cheaper). **Caveat**: this session implemented the
  architecture as described here (frozen encoder + gated bounded memory), not a
  line-by-line port re-derived from the paper's own equations -- see the module's
  docstring.
- **METER** ([arXiv:2312.16831](https://arxiv.org/abs/2312.16831), 2023) -- a "dynamic
  concept adaptation framework," explicitly designed around detecting and adapting to
  concept drift online. Directly relevant to this project's own dynamic-threshold work
  (`RollingPercentileThreshold`/`POTThreshold`) -- METER adapts the *model*, not just the
  *threshold*, which is a different and complementary lever from what this project has
  explored so far.
- **PySAD** (Python Streaming Anomaly Detection) -- a benchmarking library specifically
  for streaming/online detectors (iForestASD, Half-Space Trees, RRCF, and others under
  one API). Worth a look purely as a benchmarking harness, similar in spirit to how this
  project already benchmarks against TimeEval-style tooling indirectly via ESA-ADB.

## 5. Satellite / spacecraft telemetry specific

- **ESA-ADB** (Kotowski et al. 2024, [arXiv:2406.17826](https://arxiv.org/abs/2406.17826))
  -- this project's own base dataset/benchmark; see main README for full treatment.
  **Its own Supplementary Table 9 ("results for anomalies alone")**, extracted this
  session from the paper's full PDF text (not just the abstract), reports corrected
  event-wise F0.5 <= 0.001 for every algorithm the authors themselves tested on
  Mission2's full channel set, real anomalies only -- including their own all-events
  champion, `Windowed iForest` (0.949 there). This is the external reference point the
  main README previously flagged as missing; see README "Anomalies only" for the direct
  comparison. Practical upshot for this literature review: anomalies-only detection on
  this dataset is near-unsolved in the published record, not a bar this project is
  falling short of.
- **OPS-SAT / OPSSAT-AD** ([arXiv:2407.04730](https://arxiv.org/abs/2407.04730),
  *Nature Scientific Data* 2025, also from KP Labs -- the same group behind ESA-ADB) --
  a NEWER satellite anomaly benchmark, from ESA's OPS-SAT CubeSat mission, with baseline
  results from 30 supervised and unsupervised algorithms. A real, natural extension
  target for this project beyond Mission1/Mission2: smaller/different mission profile
  (CubeSat vs. the larger ESA-ADB missions), and **OPS-SAT VOLT** (launching late 2025)
  is explicitly intended to let researchers validate algorithms *on board an operational
  satellite* -- i.e. a genuine path from this kind of offline benchmarking to real
  onboard deployment, worth watching.
- **"Deep Learning-Based Anomaly Detection in Spacecraft Telemetry on Edge Devices"**
  ([arXiv:2603.29375](https://arxiv.org/abs/2603.29375), 2025/2026) -- directly about
  running spacecraft anomaly detection under onboard compute constraints; the closest
  published work to this project's own "laptop-friendly, day-1-of-a-new-satellite"
  framing pushed one step further (actual flight hardware rather than ground-segment
  laptops). Worth reading in full for concrete resource-budget numbers to compare
  against this project's own (e.g. the GDN memory bugs found and fixed this session).
- **Φsat-2** (ESA, launched August 2024) -- not an anomaly detector, but a concrete,
  recent example of onboard AI actually flying: an 8-bit-quantized CNN for cloud
  detection running within CubeSat hardware limits. A useful reference point for how
  aggressive quantization/compression needs to be for genuine onboard deployment, beyond
  what this project currently targets (a laptop, not flight hardware).
- **Telemanom** (Hundman et al., KDD 2018) / **Telemanom-ESA** (ESA-ADB's own
  extension) -- already this project's `LSTMForecaster` reference point; the paper's own
  strongest performer for Mission1's full channel set (see main README).

## 5.5. Contaminated-training-data methods (the actual L2 problem)

Everything in  1-5 assumes a clean-nominal fit set. This project's own L2 numbers (main
README, "Anomalies only") show that assumption is the real bottleneck, not online
training per se -- so this is its own literature thread, reviewed and (partially)
integrated this session.

- **LOE -- Latent Outlier Exposure** (Qiu, Li, Kloft, Rudolph, Mandt, ICML 2022,
  [arXiv:2202.08088](https://arxiv.org/abs/2202.08088), code:
  [boschresearch/LatentOE-AD](https://github.com/boschresearch/LatentOE-AD)) --
  **integrated** as `models/heavy/loe_autoencoder.py` (`LOEAutoencoder`, runners
  `LOEAutoencoder-H`/`-S`). Jointly infers a binary latent label per training row each
  minibatch (the worst `contamination`-fraction by `L_n - L_a` goes to a "latent anomaly"
  group) while training two coupled losses -- one normal (`L_n`, the usual reconstruction
  loss), one anomalous (`L_a`, only ever seen by rows currently assigned anomalous).
  "Hard" LOE makes a 0/1 assignment; "soft" LOE splits assigned-anomalous rows 50/50
  between both losses to hedge against a wrong assignment. At test time only `L_n` is
  ever used, so nothing about the anomaly loss needs to be trusted at inference.
  **Caveat found while integrating**: the paper only ever defines `L_a` for
  self-supervised backbones (MHRot/NTL/ICL, all image/tabular but not plain
  reconstruction); for a plain MSE autoencoder (this project's `Autoencoder`) there's no
  official `L_a`. `models/heavy/loe_autoencoder.py` uses a bounded hinge-style substitute,
  `L_a = relu(margin - L_n)`, which is a reasonable but **not paper-validated** design
  choice for this backbone -- flagged as a caveat, not a bug. Sensitivity study in the
  paper (Fig. 3) shows both variants stay robust to a misjudged `contamination`
  hyperparameter (up to 15-20 points off); this project's Mission 2 training split is
  ~7.3% non-nominal rows against the code's 10% default, comfortably inside that robust
  range. **Result, corrected this session (see 5.6)**: neither LOE variant beats the
  blind `Autoencoder` baseline once measured under a reproducible rerun -- the earlier
  "LOE-hard wins" finding was an artifact of the CPU-thread nondeterminism below, not a
  real effect. Numbers in main README.
- **USDR -- Unsupervised Data Refinement** (Ulmer, Zgraggen, Goren Huber, ZHAW,
  [arXiv:2308.13352](https://arxiv.org/abs/2308.13352)) -- **integrated** as
  `models/refinement.py` (`USDRRefiner`, runner `USDR-SubsystemPCAEnsemble`). Trains `M`
  residual-based models on overlapping row subsets, then scores each row by (mean
  residual under models that *excluded* it) minus (mean residual under models that
  *included* it) -- a row that reconstructs worse when left out of training is
  suspicious. No contamination-rate hyperparameter, model-agnostic, single-step (not
  iterative like LOE). This project's `USDRRefiner.scores_from_residuals` is a direct,
  confirmed match to the paper's excluded-minus-included formulation, with two
  simplifications: random per-row subset membership instead of the paper's time-ordered
  sliding-window subsets (built for contextual/temporal anomalies -- less relevant here
  since the downstream `SubsystemPCAEnsemble` treats rows independently), and a
  self-added MAD-based rejection cutoff (the paper deliberately leaves thresholding to
  "a separate study"). Confirmed bit-reproducible across repeated runs (doesn't use
  PyTorch, unaffected by 5.6) -- README numbers stand unchanged.
- **MemStream** ([arXiv:2106.03837](https://arxiv.org/abs/2106.03837), Bhatia et al.,
  WWW 2022, see 4) -- **integrated** as `models/heavy/memstream.py` (runner
  `MemStream`). A third approach to the same L2 problem: instead of reweighting the
  loss (LOE) or rejecting rows before fitting (USDR), freeze the encoder after one
  pretrain and gate a bounded online memory to only ever admit rows that already score
  as normal -- contamination tolerance without a `contamination` hyperparameter or a
  rejection step. Doesn't beat blind `Autoencoder` in this project's L2 protocol
  (main README); the `beta_` admission cutoff is itself calibrated from the (already
  contaminated, at L2) fit batch, which likely blunts the gating mechanism's edge. Its
  confirmed, genuine advantage is cost, not accuracy: no gradient step means
  `partial_fit` is markedly cheaper than every other heavy model here.
- **CLEANet / CRTF** (Zhang et al., 2025, [arXiv:2510.22619](https://arxiv.org/abs/2510.22619))
  -- **integrated** as `models/heavy/crtf_autoencoder.py` (runner `CRTFAutoencoder`). A
  fourth approach, this one layered onto the plain `Autoencoder` rather than a separate
  backbone: the paper's "Contamination-Resilient Training Framework" combines an
  adaptive reconstruction weighting strategy with clustering-guided contrastive
  learning; here that's a per-epoch k-means over the model's own current latent space
  plus a continuous residual-based sigmoid weight (no fixed contamination rate, unlike
  LOE) that down-weights both the reconstruction loss and a cluster-centroid
  contrastive term for high-residual rows. **Result (main README), confirmed across 2
  seeds (42, 123)**: the first of the four contamination-robust techniques tried this
  session that doesn't just lose outright -- beats blind `Autoencoder` on pooled
  anomalies-only F0.5 at every budget in both seeds (clearest at 10/mo: same or better
  recall with roughly a third of the false alarms). A second, arguably bigger finding
  fell out of the seed check: blind `Autoencoder`'s own best-budget pooled F0.5 swings
  >15x between seeds (both runs individually bit-reproducible -- genuine seed variance,
  not the 5.6 threading issue), while CRTF's swings only ~25%, itself indirect evidence
  the mechanism is doing something real rather than winning by luck. Still only 2 seeds
  on one dataset (Mission 2) -- a paper-split cross-check and Mission 1 remain open.
  **Caveat**: not a verified port -- only the paper's abstract
  was readable this session (full-text PDF extraction failed), so the two named
  mechanisms are implemented as a documented, good-faith interpretation of what those
  names describe, not equations re-derived from the source; see the module's own
  docstring.
- **Encode-then-Decompose (EDAD)** (Zhang, Kieu, Qiu, Guo, Hu, Zhou, Jensen, Yang,
  [arXiv:2510.18998](https://arxiv.org/abs/2510.18998), ICDE 2026 extended) --
  **reviewed, not integrated.** A 3-layer attention encoder whose representation is split
  into a "stable" and an "auxiliary" component, scored via an InfoNCE-style
  mutual-information estimator instead of reconstruction error, explicitly to avoid the
  reconstruction bottleneck contamination exploits. Evaluated on GPU (A800); a
  transformer-scale backbone well outside this project's CPU-only, closed-form/PCA-scale
  design budget. Worth reconsidering only if this project ever grows a GPU-capable heavy
  tier.
- **ATHMoS** (Schlag, Schefels, Helmsauer, DLR/GSOC,
  [elib.dlr.de/196102](https://elib.dlr.de/196102/)) -- **reviewed, not integrated.** An
  operational ground-segment tool (DLR's ViDA software) producing an explainable anomaly
  probability from generic per-channel statistics; a system description, not a novel
  algorithm. No new technique to port, but validates this project's own bias toward
  simple, explainable, deployable detectors (`GlobalSTD` is the closest analogue already
  here).

## 5.6. A reproducibility finding: PyTorch CPU thread count and online training

Not literature -- a methodology finding surfaced this session while trying to reproduce
the LOE comparison above, worth recording because it changes how much to trust any
small delta between PyTorch-backed models in this project's results so far. PyTorch's
CPU intra-op parallelism doesn't guarantee the same floating-point reduction order
across runs (thread scheduling, concurrent machine load); for one forward/backward pass
that's noise far below any metric's resolution, but this project's walk-forward
protocol chains dozens of `partial_fit` calls per model across folds, and iterative
gradient descent is chaotically sensitive to exactly this kind of tiny per-step
perturbation. Confirmed empirically, same seed=42: the identical `Autoencoder` run
produced anomalies-only F0.5 values 2-4x apart depending only on CPU thread count /
concurrent load, with event-level recall matching almost exactly (the model detects the
same real anomalies) while precision (how many total alarms it raised) drifted --
consistent with the mechanism, since alarm *count* under budget calibration is exactly
the kind of threshold-sensitive quantity a slightly different training trajectory would
shift. Pinning `torch.set_num_threads(1)` (now project-wide, `models/heavy/__init__.py`)
was confirmed to make a model's results identical whether it trains alone or shares the
process with another PyTorch model in the same walk-forward run -- previously it
didn't. Non-PyTorch models (`USDR-SubsystemPCAEnsemble`, `GlobalSTD`, PyOD wrappers,
etc.) were checked and are unaffected. Practical upshot for reading this project's own
numbers: any comparison between PyTorch-backed models (`Autoencoder`,
`LOEAutoencoder-H`/`-S`, `LSTMForecaster`, `MemStream`, `GDN`) generated before this fix
isn't guaranteed reproducible, and small deltas (single-digit-percent) between them
should be treated as within noise unless regenerated under the pinned-thread code.

**A second, related finding from the CRTF seed check (5.5)**: even with the threading
fix applied -- i.e. each individual run now fully bit-reproducible -- blind
`Autoencoder`'s best-budget pooled anomalies-only F0.5 still swings **>15x between
random seeds 42 and 123** (0.084 -> 0.005). This is genuine seed variance, not
nondeterminism: a `torch.manual_seed`-driven weight initialization and shuffle order
that happens to land in a better or worse basin, amplified by how few real-anomaly
events this metric has to average over. `CRTFAutoencoder` swung only ~25% on the same
check, which is itself indirect evidence its contamination-resilience mechanism is
real rather than a lucky draw. Upshot: even a *reproducible* single-seed number for a
PyTorch-backed model's anomalies-only F0.5 in this project should be read as one sample
from a wide distribution, not a point estimate -- multi-seed reporting (as the README's
`Autoencoder-replay` all-events seed check already does for a different metric) is
worth extending to every anomalies-only PyTorch comparison, not just CRTF's.

## 6. Lightweight / edge-deployable models (non-satellite, but directly relevant technique-wise)

- Quantized TinyML models (~0.92 F1, 3x memory reduction, 60% lower energy vs. larger
  models in one 2025 IoT sensor study) -- confirms quantization as a viable, large lever
  not yet explored in this project (all models here run at full float32/float64
  precision).
- **TransEdge** (2025/2026, *Journal of Intelligent Information Systems*) -- a
  lightweight transformer specifically compressed for edge anomaly detection; relevant
  as a "what would a leaner Anomaly Transformer look like" reference if this project
  ever explores transformer-based detection (not currently in the model lineup).
- **Matrix Profile methods** (DAMP, MADRID) -- already flagged as rejected-but-promising
  in the ESA-ADB paper itself (fast, interpretable, can theoretically memorize rare
  nominal events) and independently worth reconsidering here given this project's own
  `RollingPercentileThreshold`-style emphasis on lightweight, lookup-based methods over
  heavier learned ones.

## 7. Where this project sits relative to the above

| Property | This project | Most cited SOTA above |
|---|---|---|
| Online training (OT) | Yes -- `partial_fit()` on 5 of 7 models, validated on real Mission2 data | Rare; explicitly flagged as missing for autoencoders in the 2024 survey |
| Online inference (OI) | Yes -- chunked `score()`, walk-forward monthly folds | Common for RNN/forecasting models, rare for reconstruction models |
| Unsupervised/DwL | Yes -- every model fits on nominal-only data | Standard for the field |
| Explainability | Yes -- `score_per_channel`/`most_anomalous_channels` on 5 models, plus `GDN`'s learned graph | Uncommon outside GDN-style graph methods and a handful of autoencoder root-cause variants ( 6.3 of the online survey) |
| Satellite-specific validation | Yes -- real Mission1/Mission2 data, paper-metric-replicated | ESA-ADB paper, OPS-SAT, Telemanom lineage only |
| Lightweight/quantized | No -- full-precision throughout | An active, separate research thread ( 6) not yet touched here |
| Foundation-model / zero-shot | No | The newest, most active 2025-2026 direction ( 3) |

## 8. Concrete next steps this review suggests

1. ~~Read MemStream in full before any further online-autoencoder design work~~ --
   **done this session**: implemented as `models/heavy/memstream.py`, see 5.5. Didn't
   beat blind `Autoencoder` on accuracy here, but confirmed genuinely cheaper online
   updates (no gradient step) -- worth another look if a future backbone's `beta_`
   admission cutoff can be calibrated on less-contaminated data than the raw L2 fit
   batch.
2. **Try Online Isolation Forest** ([arXiv:2505.09593](https://arxiv.org/html/2505.09593v1))
   as a more principled alternative to `WindowedIsolationForest`'s current
   `warm_start`-based `partial_fit`.
3. **Try Robust Random Cut Forest** alongside `HalfSpaceTrees` in the "Open next steps"
   online-tree-model comparison already planned in the main README.
4. **Consider `river`** as a dependency for any future streaming-tree model, rather than
   continuing to hand-roll them (this project already hand-rolled `HalfSpaceTrees`, with
   real, confirmed performance bugs along the way -- a mature library might avoid that
   class of issue for the *next* streaming-tree model tried).
5. **Quantization is untried here and has a real, cited payoff elsewhere** (~3x memory,
   ~60% energy) -- a natural fit for this project's own "laptop-friendly" and "day 1 of
   a newly launched satellite" (i.e. eventually onboard) framing, not yet explored.
6. **OPS-SAT/OPSSAT-AD** is a natural extension dataset once Mission1/Mission2 work is
   further along -- same team as ESA-ADB, a genuinely different (CubeSat) mission
   profile, and an explicit path (OPS-SAT VOLT) to real onboard validation.

## Sources

- [Correia et al., "Online Model-based Anomaly Detection in Multivariate Time Series"](https://arxiv.org/abs/2408.03747) (2024)
- [Deng & Hooi, "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series"](https://arxiv.org/abs/2106.06947) (AAAI 2021) -- this project's `GDN`
- [Xu et al., "Anomaly Transformer"](https://arxiv.org/abs/2110.02642) (ICLR 2022)
- [MAAT: Mamba Adaptive Anomaly Transformer](https://arxiv.org/html/2502.07858v1) (2025)
- [Tuli et al., "TranAD"](https://arxiv.org/abs/2201.07284) (VLDB 2022)
- [Bhatia et al., "MemStream"](https://arxiv.org/abs/2106.03837) (WWW 2022) -- this project's `MemStream`
- [Zhang et al., "CLEANet: Robust and Efficient Anomaly Detection in Contaminated Multivariate Time Series"](https://arxiv.org/abs/2510.22619) (2025) -- this project's `CRTFAutoencoder`
- [METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection](https://arxiv.org/abs/2312.16831) (2023)
- [Guha et al., "Robust Random Cut Forest Based Anomaly Detection on Streams"](https://proceedings.mlr.press/v48/guha16.pdf) (ICML 2016)
- ["Online Isolation Forest"](https://arxiv.org/html/2505.09593v1) (2025)
- [Kotowski et al., "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry"](https://arxiv.org/abs/2406.17826) (2024) -- this project's base
- [The OPS-SAT benchmark for detecting anomalies in satellite telemetry](https://arxiv.org/abs/2407.04730) (2025)
- ["Deep Learning-Based Anomaly Detection in Spacecraft Telemetry on Edge Devices"](https://arxiv.org/abs/2603.29375) (2025/2026)
- [Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"](https://dl.acm.org/doi/10.1145/3219819.3219845) (KDD 2018) -- Telemanom, this project's `LSTMForecaster` basis
- ["Towards Foundation Models for Zero-Shot Time Series Anomaly Detection"](https://arxiv.org/abs/2509.21190) (2025)
- [TimeRadar: A Domain-Rotatable Foundation Model for Time Series Anomaly Detection](https://arxiv.org/pdf/2602.19068) (KDD 2026)
- [Qiu et al., "Latent Outlier Exposure for Anomaly Detection with Contaminated Data"](https://arxiv.org/abs/2202.08088) (ICML 2022) -- this project's `LOEAutoencoder`
- [Ulmer, Zgraggen, Goren Huber, "A Generic Machine Learning Framework for Fully-Unsupervised Anomaly Detection with Contaminated Data"](https://arxiv.org/abs/2308.13352) (2023) -- this project's `USDRRefiner`
- [Zhang et al., "An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection on Contaminated Training Data"](https://arxiv.org/abs/2510.18998) (ICDE 2026 extended)
- [Schlag, Schefels, Helmsauer, "Applying Machine Learning to Routine Satellite Ground Segment Operations by Means of Automated Anomaly Detection"](https://elib.dlr.de/196102/) (DLR/GSOC)
