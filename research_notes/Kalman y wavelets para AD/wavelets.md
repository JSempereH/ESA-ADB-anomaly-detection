# Wavelet-Based Multivariate Anomaly Detection for Streaming Satellite Telemetry

Scope: feasibility assessment of discrete/continuous wavelet transform (DWT/WPT/CWT) and
multi-resolution-analysis methods for multivariate, contaminated, online anomaly detection,
assessed against an already-strong baseline (Kalman local-trend F0.5=0.070, Matrix Profile
discord detection F0.5=0.074) on ESA-ADB Mission 1/2 telemetry. Fourier/Spectral-Residual
methods are explicitly out of scope (already researched and rejected).

## 1. What is the SOTA (2023-2026) for wavelet-based anomaly detection on MULTIVARIATE time series?

### Takeaway
There is no mature, widely-adopted 2023-2026 wavelet-specific SOTA for multivariate anomaly
detection comparable to Matrix Profile or deep contamination-robust autoencoders; the field is
fragmented into (a) wavelets-as-a-feature-preprocessing-step feeding a conventional detector
(isolation forest, CNN, autoencoder) and (b) a small, older, still-unmatched line of genuinely
online wavelet+statistical-model work (Thill, Konen & Bäck) that itself is univariate/per-series
rather than cross-channel-joint.

### Cited Findings
- WMAD (Wavelet transform for Multiscale time series Anomaly Detection), Ye & He, *Journal of
  Computer Applications* Vol. 44 Issue 10, Oct 2024 (DOI 10.11772/j.issn.1001-9081.2023101480):
  wavelet-decomposes a series into frequency components, stacks multi-temporal 2D windows, and
  feeds a multiscale CNN with spatial+channel attention. Chinese-language paper; abstract only
  reviewed. — [joca.cn](https://www.joca.cn/CN/Y2024/V44/I10/3300)
- CRAFTIIF (Cross-Resolution Analytic Four-Type Interpretable Isolation Forest), arXiv:2606.13486,
  single-author preprint (William Smits), posted 2026: generates K=500 random wavelet features
  across four wavelet families (Morlet, DOG, Haar, Coiflet) each targeted at a different anomaly
  type (point/distributional/temporal/collective), feeds five structured Isolation Forests (one
  per type + a meta-IF), with an adaptive Otsu/MAD threshold. Evaluated on all 19 mTSBench
  datasets: mean F1=0.228 (all 19), F1=0.322 (13 "detectable" datasets), VUS-PR=0.463 vs. prior
  best 0.329 (+40.7%), claims 6/19 datasets "fundamentally undetectable" by any unsupervised
  method. The paper's own AI-use statement discloses LLM (Claude) assistance in writing the
  manuscript, and no confirmed public code repository was found. Treat as unreplicated/low-
  confidence. — [arxiv.org/pdf/2606.13486](https://arxiv.org/pdf/2606.13486), [sotaverified.org listing](https://sotaverified.org/papers/260613486)
- Online DWT + multivariate Gaussian anomaly detector, Thill, Konen & Bäck (~2018, KIT
  repository): iteratively computes a **causal, decimating DWT** and fits a multivariate Gaussian
  per frequency scale online; flags anomalies from cross-scale pattern deviation. Tested on 425
  time series — the count exactly matches NAB (58 series) + Yahoo Webscope S5 (367 series),
  strongly suggesting those were the benchmarks (not independently confirmed from the primary
  text, which could not be extracted as readable text). — [KIT repository record](https://publikationen.bibliothek.kit.edu/1000097489/37179700)
- DWT-MLEAD (the same research group's earlier/related offline method, Haar wavelets +
  Mahalanobis distance across scales) has been reimplemented (with noted deviations from the
  original paper — no Monte Carlo quantile boundary, no anomaly-clustering step) in the
  maintained `aeon-toolkit` and `dtaianomaly` Python benchmarking libraries as a **univariate**
  detector. — [aeon docs](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.anomaly_detection.series.distribution_based.DWT_MLEAD.html), [dtaianomaly source](https://dtaianomaly.readthedocs.io/en/latest/_modules/dtaianomaly/anomaly_detection/_DWT_MLEAD.html)
- Graph-spectral + wavelet-packet fusion for industrial IoT networks, arXiv 2026 preprint:
  applies a graph Fourier transform across the sensor network first, then a Daubechies-4 wavelet
  packet transform (depth 3, 8 sub-bands) per graph-frequency component, using sub-band energies
  as features; explicitly a batch feature pipeline, not demonstrated online. —
  [arxiv.org/pdf/2605.03591](https://arxiv.org/pdf/2605.03591)
- Wavelet-packet + higher-order-statistics "noise signatures" for anomaly prediction, arXiv
  2512.12528 (late 2025): wavelet-packet-splits a fused sensor signal into structure/residual,
  computes third-order cumulants on the residual, detects via Mahalanobis distance with a
  closed-form threshold; framed for wireless/fused sensing, theory-heavy, no telemetry
  validation. — [arxiv.org/html/2512.12528v1](https://arxiv.org/html/2512.12528v1)

### Inferences
- The strongest, most carefully engineered wavelet+ML papers found (WMAD, CRAFTIIF, graph-
  wavelet-packet) all route the wavelet decomposition into a conventional learned detector
  (CNN, Isolation Forest) rather than demonstrating wavelets' own time-frequency localization as
  the primary mechanism — meaning the project would effectively be re-implementing "IsolationForest
  on richer features," not a fundamentally new detection paradigm, and PyOD's IForest/ECOD/COPOD
  are already in the benchmark.
- The only line of work that treats the wavelet transform itself (not as a feature extractor for
  a downstream learner) as the anomaly-scoring mechanism is the decade-old Thill/Konen/Bäck
  family, and it is univariate per series (one Gaussian per scale per series), requiring a
  separate cross-channel fusion layer to become genuinely multivariate — which is exactly the
  "per-channel decomposition + fusion" pattern in Key Question 2, not a built-in joint method.

### Gaps
- Could not retrieve full text of the WMAD paper (Chinese-language, abstract-only access) to
  confirm whether it handles contamination or reports wall-clock cost.
- Could not independently verify CRAFTIIF's reported numbers (no peer review found, no confirmed
  code release, single author with disclosed LLM-assisted writing) — numbers above should be
  treated as unverified claims, not established fact.

## 2. Is there a well-established MULTIVARIATE extension (cross-channel wavelet coherence / cross-wavelet transform)?

### Takeaway
Yes, a well-established statistical machinery exists for genuinely joint multi-channel wavelet
analysis — wavelet coherence / cross-wavelet transform, and its multivariate "locally stationary
wavelet" (LSW) generalization with a wavelet coherence matrix — and it has been applied to detect
exactly the failure mode described in the question (breakdown of the normal time-frequency
relationship between channels) on real industrial multi-channel sensor data. However, the one
concrete, validated implementation found is a **supervised classifier** (requires labeled
training signals per class), which conflicts with this project's zero-labels constraint.

### Cited Findings
- Foundational cross-wavelet/coherence theory: Grinsted, Moore & Jevrejeva (2004, geophysical
  series) and Torrence & Compo's practical guide — cross-wavelet transform = product of one
  series' wavelet transform with the conjugate of another's (shared energy by time+scale);
  coherence normalizes this by each series' individual power. — [metricgate.com cross-wavelet-transform](https://metricgate.com/docs/cross-wavelet-transform), [pyleoclim wtc docs](https://pyleoclim-util.readthedocs.io/en/v0.7.4/utils/wavelet/wtc.html)
- Wilson, Eckley, Nunes & Park, "Dynamic detection of anomalous regions within distributed
  acoustic sensing data streams using locally stationary wavelet time series," arXiv:1809.09729
  (2018). This is the most directly relevant paper found for Key Questions 2+3+4 combined:
  - **Real industrial multivariate sensor data**: Distributed Acoustic Sensing (DAS) from an
    actual oil/gas well, fiber-optic sensing at multiple depths — genuinely multi-channel
    telemetry, not a synthetic benchmark.
  - **Genuinely joint cross-channel method**: builds a multivariate locally-stationary-wavelet
    (MvLSW) model with a wavelet coherence matrix ρ_jk(z) measuring dependence between channel
    pairs at a given time and scale; the target anomaly ("striping") is defined precisely as a
    *simultaneous, non-physical change appearing across multiple channels/depths at once* — i.e.
    a break in the normal cross-channel time-frequency relationship, exactly matching the
    question's framing. — [arxiv.org/pdf/1809.09729](https://arxiv.org/pdf/1809.09729) (full text extracted via pdftotext)
  - **It is a supervised classifier, not an unsupervised detector**: it requires a set of
    training signals Y^(i) with *known class membership* C_Y(i)(t) over time, from which the
    Fisher-z-transformed coherence mean/variance per class are estimated; classification of new
    signals uses Bayes' rule against these class-conditional Gaussians (paper Section 2,
    equations 4-6). No label-free variant is described. — same source, lines ~560-630 of
    extracted text.
  - Online updating exists: "the online wavelet transform is then used to efficiently update the
    wavelet coefficients and the transformed coherence estimate for the new window," reusing
    prior calculations rather than fully batch-recomputing. — same source, lines ~672-738.
- A separate 2018 arXiv paper generalizes wavelet coherence to multivariate, non-stationary
  series using surrogate-data significance testing for transient coherent patterns (not
  demonstrated as an anomaly detector per se). — [ar5iv.arxiv.org/html/1801.04778](https://ar5iv.arxiv.org/html/1801.04778)
- "Coherent Multiplex" (arXiv:2508.19994, 2025): a graph-based framework combining spectral
  similarity filtering with wavelet coherence for large multivariate series; explicitly states
  that significance testing, adaptive thresholding, and ML integration are future work — i.e.
  not a finished anomaly detector. — [arxiv.org/pdf/2508.19994](https://arxiv.org/pdf/2508.19994)
- Software for wavelet coherence exists and is usable: Grinsted's MATLAB code, `pycwt` (Python),
  `WaveletComp` (R, with surrogate-based coherence significance), `pyleoclim.utils.wavelet.wtc`.
  None of these packages include an anomaly-detection layer on top — they compute coherence only.
  — [pyleoclim docs](https://pyleoclim-util.readthedocs.io/en/v0.7.4/utils/wavelet/wtc.html)

### Inferences
- The coherence-matrix idea is directly transferable in spirit to satellite telemetry (e.g.
  detecting when two normally-correlated channels, such as a heater command and the temperature
  it drives, decouple at a particular time-scale) — but the only validated real-data
  implementation is supervised, so it would need to be re-derived as an unsupervised/self-
  referential baseline (e.g., compare live coherence against a rolling "normal" coherence
  distribution estimated online, analogous to how Kalman/Matrix Profile build their own
  reference without labels) rather than borrowed wholesale.
- A pairwise coherence matrix across 47-58 channels is O(C²) channel pairs (~1,100-1,650 pairs
  for Mission 1/2), which is a non-trivial but not prohibitive blow-up versus the existing
  per-channel detectors; this is the main structural cost risk of pursuing the "genuinely joint"
  variant rather than a simpler per-channel-decomposition-plus-fusion variant.

### Gaps
- No peer-reviewed, 2023-2026, real-data paper found that performs wavelet-coherence-based
  anomaly detection in a fully unsupervised/label-free manner (all real-data validations found
  are supervised classification).
- Did not find a direct comparison of wavelet-coherence detection cost/accuracy vs. the simpler
  "per-channel wavelet decomposition + late fusion" alternative on the same dataset.

## 3. Is it usable ONLINE/incrementally? What's the realistic compute cost for 47-58 channels at 148k-440k rows/month?

### Takeaway
Classic textbook DWT is a batch, power-of-two block operation, but genuinely incremental/causal
wavelet decompositions exist and have been demonstrated online (Thill/Konen/Bäck; Wilson et al.
DAS paper) with near-constant per-step cost; separately, even naive full-window batch recompute
of a DWT via PyWavelets is numerically cheap enough (sub-millisecond per channel for windows of
hundreds-to-low-thousands of samples) that it would likely be acceptable for this project's scale
(47-58 channels, 148k-440k rows/month) without needing the incremental variant at all — though no
source gives concrete wall-clock numbers at this exact scale, so this is an engineering inference,
not a cited benchmark.

### Cited Findings
- Thill, Konen & Bäck's online detector performs a **causal, decimating DWT** update step-by-step
  rather than full-window batch recomputation, and a related chapter explicitly attributes
  computational efficiency to "the linear time complexity of the DWT." — [KIT repository](https://publikationen.bibliothek.kit.edu/1000097489/37179700), [Leiden University thesis repository](https://scholarlypublications.universiteitleiden.nl/access/item%3A3279169/view)
- Wilson et al. (DAS paper, arXiv:1809.09729): "the online wavelet transform is then used to
  efficiently update the wavelet coefficients and the transformed coherence estimate for the new
  window... reusing data calculations" (i.e., avoids recomputing coefficients from scratch each
  step). A runtime-scaling experiment (series lengths T=1024/2048/4096/8192, 25 replications) found
  their online method has "near-constant scaling with the length of the series" after an initial
  ramp-up, second-best among compared online classifiers — but the paper reports this only as a
  qualitative scaling plot (Figure 5), with **no concrete wall-clock/throughput numbers** in the
  extracted text. — [arxiv.org/pdf/1809.09729](https://arxiv.org/pdf/1809.09729)
- A separate streaming-sensor-network line of work argues the standard dyadic (power-of-two)
  Mallat DWT decomposition is itself limiting for streaming outlier detection and proposes
  non-dyadic Haar wavelets as an alternative better suited to arbitrary streaming window
  lengths (2010, pre-2023, background only). — search result summary, non-dyadic Haar wavelets paper (Stevens Institute)
  — [researchwith.stevens.edu](https://researchwith.stevens.edu/en/publications/non-dyadic-haar-wavelets-for-streaming-and-sensor-data/)
- PyWavelets itself (the standard Python DWT/WPT library) performs only batch forward/inverse
  transforms — it has no incremental/streaming API; any online use requires sliding-window
  re-invocation of the library by the caller. — search synthesis, [PyWavelets docs](https://pywavelets.readthedocs.io/en/v0.4.0/index.html)
- A "Wavelet-GARCH" live-monitoring report (NIH ORF technical resources, May 2026) applies
  sliding-window DWT denoising to live 15-minute-interval chilled-water flow sensor data,
  reusing the previous window's data to handle boundary artifacts at each step — i.e. a
  practical sliding-window batch-recompute design, not a mathematically incremental transform,
  used successfully for real streaming sensor monitoring. — [NIH ORF PDF](https://orf.od.nih.gov/TechnicalResources/Documents/News%20to%20Use%20PDF%20Files/2026%20NTU/Denoising%20Data%20-%20Utilizing%20Wavelet-GARCH%20Technique%20for%20Anomaly%20Detection%20-%20May%202026%20NTU_508.pdf)

### Inferences
- For this project's walk-forward protocol, the realistic engineering path is **sliding-window
  batch recompute via PyWavelets** at each step (as the Wavelet-GARCH report does), not
  re-deriving the causal/decimating DWT math from the 2018 papers — the latter would add
  substantial implementation complexity for a benefit (true incrementality) that may not be
  computationally necessary at this data rate.
- Mission 2 at 18s resolution and ~47 channels: a day of data is ~4,800 samples/channel; a
  walk-forward step recomputing a window of e.g. 512-2048 samples via `pywt.wavedec` across 47
  channels is a trivial operation for NumPy/PyWavelets (this class of operation routinely runs
  in low single-digit milliseconds per channel on commodity CPUs) — but no source in this
  research directly benchmarks PyWavelets at this exact window/channel count, so treat the
  "CPU-feasible" conclusion as a confident engineering inference rather than a cited fact.

### Gaps
- No source gives a concrete wall-clock/throughput number (e.g. "X ms per channel per window,"
  "Y channels/sec") for DWT or WPT at production telemetry scale (tens of channels, hundreds of
  thousands of rows/month) to cite directly — this would need to be established by a quick local
  microbenchmark rather than literature.

## 4. Has wavelet-based anomaly detection been validated on REAL spacecraft/satellite telemetry or comparable real multivariate industrial sensor data (2023-2026)?

### Takeaway
Real-data validation exists but is thin, older, or not peer-reviewed/English-language-confirmed:
one Russian-language ISS telemetry study and one 2026 patent are the only spacecraft-specific
sources found; the strongest real-data validation overall is the DAS oil/gas well paper (industrial,
not spacecraft, 2018). No 2023-2026 peer-reviewed paper was found applying wavelet methods to
housekeeping telemetry, power systems, or comparable continuous multivariate sensor streams with
a rigorous accuracy benchmark.

### Cited Findings
- A Russian engineering-journal paper describes using wavelet transforms (with a described
  step-by-step wavelet filtering process) to automate anomaly detection in spacecraft onboard
  systems, tested on **archival telemetry from the ISS Russian Segment's air-conditioning and
  power-supply systems** — i.e. real spacecraft housekeeping telemetry, directly analogous to
  ESA-ADB. Only the English abstract/summary was reviewed, not the full Russian text, so method
  details and quantitative results are not independently confirmed. — [engjournal.bmstu.ru (onboard systems diagnosis)](https://engjournal.bmstu.ru/eng/catalog/arse/ahtp/1890.html)
- US Patent 12,093,804 (2026 grant), "Satellite anomaly detection method and system for
  adversarial network auto-encoder": pairs a wavelet-analysis step with an adversarial
  autoencoder; the wavelet step computes **wavelet variance** of satellite telemetry over a
  backward period window to determine an error threshold at each moment, using a 97% upper
  quantile of that backward window — i.e., an unsupervised, training-label-free quantile
  threshold directly on wavelet-variance, structurally similar to how the project's existing
  Kalman/Matrix Profile detectors calibrate thresholds without labels. — [USPTO patent 12093804](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12093804)
- Wilson et al. DAS paper (arXiv:1809.09729, 2018): real industrial multivariate sensor data
  (oil/gas well fiber-optic sensing), but not spacecraft, and supervised (see Key Question 2).
- A 2026 arXiv paper on edge-deployable spacecraft anomaly detection notes space-qualified
  systems must fit tight power budgets and withstand radiation, and benchmarks
  forecasting-with-threshold / classification / image-based approaches on the ESA Anomaly
  Dataset (ESA-ADB's own lineage), reporting forecasting+threshold achieved 92.7% CEF0.5 — this
  paper does **not** use wavelets, but is relevant context that non-wavelet forecasting methods
  (like this project's own Kalman/GBM) are competitive/state-of-the-art on this exact dataset
  family. — [arxiv.org/html/2603.29375v1](https://arxiv.org/html/2603.29375v1)
- A PHM Society paper on unsupervised anomaly detection for long-duration robotic-mission
  telemetry lists wavelet software among its references but its abstract does not describe a
  wavelet-based method itself — i.e., wavelets appear only as a cited tool, not the paper's
  contribution. — [papers.phmsociety.org](https://papers.phmsociety.org/index.php/ijphm/article/view/2467)

### Inferences
- The satellite-specific evidence base for wavelets is weak relative to the strength of evidence
  already available (and already implemented) for Kalman filtering and Matrix Profile on this
  exact dataset family — no wavelet paper found reports a benchmark-comparable metric (F1,
  F0.5, VUS-PR) on ESA-ADB or a closely analogous spacecraft dataset.
- The patent's wavelet-variance + backward-window quantile design (unsupervised, training-free
  threshold) is the most directly reusable template found for this project's zero-label
  constraint, more so than any of the academic multivariate-coherence or CNN-based papers.

### Gaps
- Could not access full text of the ISS Russian Segment wavelet paper (language barrier); no
  quantitative detection metrics from it were confirmed.
- No 2023-2026 paper found benchmarking wavelet methods against modern deep or isolation-forest
  baselines specifically on spacecraft/satellite telemetry.

## 5. Is there a mature, actively maintained Python library for wavelet-based anomaly detection? If building from scratch, what's the effort estimate?

### Takeaway
No mature, actively-maintained, anomaly-detection-specific wavelet library exists; PyWavelets is
the solid, maintained building block but supplies only the transform, not scoring logic. The
closest ready-made detector (`DWT_MLEAD`) is univariate, a simplified/incomplete reimplementation
of a decade-old paper, and lives inside general time-series-anomaly benchmarking libraries
(`aeon-toolkit`, `dtaianomaly`) rather than being wavelet-anomaly-detection-focused software.
Building a per-channel wavelet-reconstruction-error/energy detector with cross-channel fusion from
PyWavelets is realistically a 4-8 hour task given the project's existing detector/fusion
infrastructure.

### Cited Findings
- PyWavelets (`PyWavelets/pywt` on GitHub, originally from a 2006 master's thesis, actively
  maintained by a dedicated team since 2013, packaged by Debian/Ubuntu) provides 1D/2D forward
  and inverse DWT, stationary wavelet transform, and wavelet packet decomposition — but no
  anomaly-scoring logic; "you would still need to write the detection logic yourself, such as
  thresholding detail coefficients." — [PyWavelets GitHub](https://github.com/nigma/pywt), [PyWavelets docs](https://pywavelets.readthedocs.io/en/v0.4.0/index.html)
- `DWT_MLEAD` is implemented in `aeon-toolkit` (class `aeon.anomaly_detection.series.distribution_based.DWT_MLEAD`) and in `dtaianomaly`. Both are **univariate** detectors (score one series at a
  time), and both libraries' own documentation discloses they deviate from the original paper:
  window sizes shrink with level, the highest decomposition level is excluded, the paper's
  Monte-Carlo quantile boundary is not implemented, and the original anomaly-clustering
  post-processing step is missing. — [aeon docs](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.anomaly_detection.series.distribution_based.DWT_MLEAD.html), [dtaianomaly source](https://dtaianomaly.readthedocs.io/en/latest/_modules/dtaianomaly/anomaly_detection/_DWT_MLEAD.html)
- No GitHub repository from the original Thill/Konen/Bäck authors was found; their method is
  only available via these third-party reimplementations.
- Smaller/less-established candidates surfaced but not verified for maintenance status:
  `anomsmith` (lists a `WaveletDetector`/`WaveletDenoiser`), `vitalDSP` (a `detect_wavelet` method
  defaulting to Haar wavelets, aimed at physiological/vital-sign signals, not general
  telemetry). — search synthesis

### Inferences
- Given `DWT_MLEAD`'s univariate design and documented deviations from its source paper, it is
  not a drop-in multivariate solution — it would need to be run once per channel (46-58 separate
  detector instances) and then fused, which is architecturally identical to what would be built
  from scratch on raw PyWavelets anyway. There is no accuracy/engineering advantage to adopting
  it over writing a custom detector, other than reusing its window/quantile boilerplate.
- Effort estimate for a from-scratch PyWavelets-based detector, informed by the project's
  existing detector patterns (GBM forecast-residual, Kalman local-trend already provide the
  windowing/scoring/fusion scaffolding to reuse):
  - Per-channel multilevel DWT (e.g. `pywt.wavedec`, Daubechies or Haar, 3-5 levels) over a
    sliding window, reconstruction-error or detail-coefficient-energy scoring, unsupervised
    rolling-quantile threshold (no labels, same pattern as existing detectors): **~2-4 hours**.
  - Cross-channel fusion via the project's existing rank-fusion ensemble machinery: **~1-2 hours**
    (mostly plumbing, since the ensemble already exists).
  - Validation/walk-forward wiring + a first benchmark run against Kalman/Matrix Profile: **~1-2
    hours**.
  - **Total: roughly 4-8 hours** for a working first version; this is an engineering estimate
    based on the project's own architecture, not a cited external source.

### Gaps
- No source directly benchmarks implementation effort for this exact task; the hour estimate
  above is inference from the project's existing codebase patterns, not from literature.

## 6. How does wavelet-based detection compare empirically against Matrix Profile / Kalman / isolation-forest-style methods on CONTAMINATED training data? Does it have a learned training phase at all?

### Takeaway
No shared-protocol benchmark comparing wavelet-based detection against Matrix Profile, Kalman
filtering, or Isolation Forest specifically under training-set contamination was found. Whether a
wavelet method has a contamination-vulnerable training phase depends entirely on which flavor is
used: simple per-channel wavelet-variance/energy threshold methods (e.g., the satellite patent's
97% backward-window quantile) are training-free/online like Matrix Profile and are therefore
structurally robust to contamination in the same qualitative way; by contrast, every
higher-performing wavelet method found in this research (WMAD's CNN, CRAFTIIF's Isolation
Forests, the DAS coherence classifier's per-class Gaussians) has a fitted/learned component that
contamination (or, for the DAS method, missing labels) would directly compromise.

### Cited Findings
- No source returned a benchmark with wavelet methods alongside Matrix Profile and Isolation
  Forest under varying contamination rates. A 2025 cow-activity-monitoring paper feeds
  wavelet-derived features into an Isolation Forest, but reports no contamination-robustness
  analysis and does not include Matrix Profile. — [arxiv.org/pdf/2502.21051](https://arxiv.org/pdf/2502.21051)
- CRAFTIIF evaluates across mTSBench's full anomaly-rate range (0.1% to 69.2% contamination in
  its "adaptive thresholding across anomaly rates" framing) but does not isolate a
  training-contamination ablation against Matrix Profile/Kalman specifically, and (as noted in
  Key Question 1) its results are unverified/unreplicated. — [arxiv.org/pdf/2606.13486](https://arxiv.org/pdf/2606.13486)
- The DAS wavelet-coherence classifier (Key Question 2) has an explicit, labeled training phase
  (per-class Fisher-z-transformed coherence Gaussians fitted from training signals with known
  class membership) — contamination of those training labels would directly corrupt the
  class-conditional distributions the whole method depends on. In a fully-unsupervised,
  zero-label setting (this project's constraint) this method cannot be used at all, independent
  of the contamination question. — [arxiv.org/pdf/1809.09729](https://arxiv.org/pdf/1809.09729)
- The satellite patent's wavelet-variance threshold (Key Question 4) recomputes its 97%-quantile
  threshold from a **backward rolling window** at each moment rather than fitting a global model
  upfront — structurally the same "training-free, reference-window-based" design as Matrix
  Profile's AB-join discord scoring, implying similar qualitative contamination behavior (an
  anomaly inside the backward window could still distort its own local threshold, exactly as
  with any training-free window-based baseline — this is a reasoned structural parallel, not a
  reported empirical result). — [USPTO patent 12093804](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12093804)

### Inferences
- If a wavelet detector is to be added to this project's benchmark on equal footing with Matrix
  Profile and Kalman (both training-free/self-referential, already validated under contamination
  in this project), it should be built in the training-free style (rolling-window
  wavelet-energy/variance quantile, per the patent's design) rather than the CNN/Isolation-Forest/
  labeled-classifier styles found in the recent literature — the latter reintroduce exactly the
  contamination vulnerability this project has spent effort avoiding with LOE/USDR/CRTF/RoCA/
  MemStream.
- No empirical head-to-head exists anywhere in the literature surveyed; any claim about wavelets
  "beating" or "losing to" Matrix Profile/Kalman under contamination would have to come from this
  project's own future experiment, not from prior publication.

### Gaps
- No contamination-stratified benchmark comparing wavelet-based methods to Matrix Profile,
  Kalman, or Isolation Forest was found in the time available. TSB-AD/TSB-UAD-style benchmarks
  were suggested by search as a likely place such a comparison could exist, but were not directly
  retrieved/confirmed to contain wavelet baselines.

---

## Overall Feasibility Verdict (synthesis across all six questions)

**Marginal / not clearly worth prioritizing next, but a cheap narrow version is plausible if
desired.** Reasoning:

1. **Genuine novelty is thinner than it first appears.** The one mechanism that is truly
   different from what's already implemented — joint cross-channel wavelet coherence detecting
   breakdown of normal inter-channel time-frequency relationships (Key Question 2) — has exactly
   one validated real-data implementation (Wilson et al., DAS wells), and it is a **supervised
   classifier requiring labeled training classes**, incompatible with this project's zero-label
   constraint. Every other "multivariate wavelet" paper found (WMAD, CRAFTIIF, graph-wavelet-
   packet) is really "wavelet features into Isolation Forest/CNN," which is not meaningfully
   different in kind from the PyOD IForest/ECOD/COPOD baselines already in the benchmark — it
   would likely land in similar performance territory for similar reasons, not open new ground.

2. **Matrix Profile already captures the stated advantage.** The core wavelet selling point for
   this use case — joint time-localization of transient anomalies that pure-frequency (Spectral
   Residual) methods smear — is exactly what Matrix Profile's discord detection already provides
   (per-channel, training-free, nearest-neighbor shape matching, already the second-best performer
   at F0.5=0.074). A simple per-channel wavelet-energy detector would likely be detecting a
   highly overlapping set of anomalies (localized shape/transient deviations) through a different
   mechanism, with no architectural reason to expect it to beat Matrix Profile rather than
   roughly replicate its catches.

3. **Real spacecraft-telemetry validation is thin.** Only one Russian-language ISS paper
   (unconfirmed quantitative results) and one 2026 patent (no peer-reviewed benchmark numbers)
   apply wavelets to real spacecraft/satellite housekeeping telemetry specifically — far weaker
   evidence than exists for Kalman filtering or Matrix Profile on this exact dataset family.

4. **Compute is not the blocker.** Unlike RRCF (rejected for being 42x slower) or HTM (rejected
   for not scaling past ~10 channels), wavelet decomposition via PyWavelets is cheap — this is the
   one dimension on which wavelets pass cleanly. If pursued, cost is not the reason to say no.

**If a decision is made to implement it anyway**, the concrete template to use is **not** any of
the 2023-2026 ML-wavelet-hybrid papers, but the **unsupervised wavelet-variance rolling-quantile
design from US Patent 12,093,804** (backward-window wavelet variance, 97th-percentile threshold,
recomputed online) combined with this project's existing rank-fusion ensemble for cross-channel
combination — built directly on PyWavelets, estimated at **4-8 hours** for a first working
version, reusing the project's existing windowing/scoring/fusion scaffolding from the GBM and
Kalman detectors. This keeps the method training-free (contamination-robust by construction, like
Matrix Profile) and avoids the labeled-classifier and learned-CNN/Isolation-Forest pitfalls found
in the rest of the literature. Expected outcome, based on the structural overlap with Matrix
Profile's time-localization mechanism, is a detector that performs in a similar range to Matrix
Profile rather than a clear improvement — making it a reasonable **ensemble-diversity addition**
(cheap, structurally different signal source for the fusion layer) rather than a standalone
breakthrough candidate.
