# Spectral Residual (SR) and Frequency/Spectral-Domain Anomaly Detection for Multivariate, Contaminated, Online Telemetry

## What exactly is Spectral Residual, and how does SR-CNN perform vs. plain SR in recent (2023-2026) comparisons?

### Takeaway
SR is a training-free, univariate, frequency-domain saliency method (FFT → log-amplitude spectral residual → inverse FFT → compare to a moving average); SR-CNN adds a CNN trained on synthetically-injected anomalies on top of the SR saliency map to turn the raw signal into a binary classifier. Both pieces come from Microsoft's original 2019 paper; no independent 2023-2026 study comparing plain SR vs. SR-CNN head-to-head was found in this search — the only comparisons available are the authors' own 2019 numbers.

### Cited Findings
- SR comes from Ren et al., "Time-Series Anomaly Detection Service at Microsoft," KDD '19 (DOI 10.1145/3292500.3330680); it is the first work to borrow the Spectral Residual model from visual saliency detection and apply it to time-series anomaly detection, and the authors combine SR with a CNN to improve on SR alone — [arXiv:1906.03821](https://arxiv.org/pdf/1906.03821).
- The SR algorithm's three steps, per Microsoft.ML's own documentation: (1) Fourier Transform to get the log amplitude spectrum, (2) calculate the spectral residual, (3) Inverse Fourier Transform back to the time domain, producing a "saliency map"; points are flagged when their saliency value deviates sufficiently from a moving average of that map — [SrCnnEntireAnomalyDetector.cs](https://source.dot.net/Microsoft.ML.TimeSeries/SrCnnEntireAnomalyDetector.cs.html); confirmed independently by Seldon's Alibi Detect docs — [Spectral Residual — Alibi Detect](https://docs.seldon.ai/alibi-detect/outlier-detection/methods/sr).
- SR alone is unsupervised and needs no training step; the CNN component is trained on synthetic (automatically-generated) anomalies, so the overall SR-CNN pipeline also avoids manual labels — [Liner summary of the paper](https://liner.com/review/timeseries-anomaly-detection-service-at-microsoft).
- For streaming/low-latency use, the paper's method appends estimated future points so the most recent real point sits near the center of the sliding window before running SR — [Liner summary](https://liner.com/review/timeseries-anomaly-detection-service-at-microsoft); corroborated by the Microsoft.ML source comments on `SrCnnEntireAnomalyDetector.cs`.
- PyOD's `SpectralResidual`/`ts_spectral_residual` module implements only the SR saliency step, explicitly not the full SR-CNN pipeline — [PyOD source](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ts_spectral_residual.html).
- The original paper's own claim is that SR-CNN "outperforms state-of-the-art baselines on both public datasets and Microsoft production data" — this is the authors' 2019 result, not an independent replication — [arXiv:1906.03821](https://arxiv.org/pdf/1906.03821).
- The TSB-AD benchmark (NeurIPS 2024 Datasets & Benchmarks, Liu & Paparrizos) includes plain "SR" as one of ~40 benchmarked algorithms, classified under "Statistical" methods, alongside IForest, HBOS, PCA, COPOD, etc. It does **not** list SR-CNN (the CNN-augmented version) as a separate benchmarked method — [TSB-AD GitHub](https://github.com/TheDatumOrg/TSB-AD), [PyPI](https://pypi.org/project/TSB-AD/).
- TSB-AD's headline finding across its 40 methods is that "simpler architectures and statistical methods often yield better performance" than advanced neural networks — a general finding, not an SR-specific one, and exact SR numeric scores vs. IForest/AutoEncoder/OmniAnomaly/AnomalyTransformer could not be retrieved from the pages fetched (the leaderboard table itself did not render in text form) — [TSB-AD leaderboard site](https://thedatumorg.github.io/TSB-AD/).

### Inferences
- Because no 2023-2026 paper directly pits SR-CNN against plain SR, the evidence base for SR-CNN's claimed improvement over SR is effectively frozen at the original 2019 authors' self-reported numbers. This is a meaningful gap for an academic related-work section — one should describe SR-CNN's advantage as "claimed by the original authors" rather than "independently confirmed."
- The fact that TSB-AD (a large, recent, independent 2024 benchmark) only includes bare SR and not SR-CNN suggests the CNN augmentation has not become a standard baseline in the broader literature — likely because it requires a trained model and synthetic-anomaly injection, undermining the "no training, fully unsupervised" selling point of SR alone.

### Gaps
- Could not obtain the actual TSB-AD numeric leaderboard scores for SR vs. IForest/AutoEncoder/OmniAnomaly/AnomalyTransformer (the fetched leaderboard page did not expose the table in text-extractable form). A follow-up fetch of the raw GitHub results CSVs (e.g., `TSB-AD/results/` or the OpenReview paper PDF table) would be needed for exact numbers.
- No independent 2023-2026 paper comparing SR vs. SR-CNN specifically was found.

---

## Is there a well-established multivariate extension of SR (per-channel SR + fusion, or a genuinely joint multivariate spectral method)?

### Takeaway
SR is fundamentally univariate by design; the only "multivariate extension" found is the engineering pattern of running SR independently per channel and fusing scores (e.g., max aggregation, as implemented in PyOD) — there is no well-established, peer-reviewed *joint* multivariate spectral-residual method. Microsoft's own multivariate production pipeline abandoned SR entirely in favor of a different architecture (MTAD-GAT, a graph-attention network), which is a strong signal that spectral residual does not extend naturally to the multivariate case.

### Cited Findings
- Alibi Detect documents SR as suited specifically to "unsupervised online anomaly detection in **univariate** time series data" — [Spectral Residual — Alibi Detect](https://docs.seldon.ai/alibi-detect/outlier-detection/methods/sr).
- PyOD's multivariate path for SR runs SR independently per channel and then aggregates the per-channel saliency scores, with max-across-channels as the default aggregation — this is a per-channel-then-fuse engineering pattern, not a joint multivariate spectral model — [PyOD source](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ts_spectral_residual.html).
- No paper proposing a genuinely joint multivariate Spectral Residual method (e.g., per-channel SR fused via cross-channel correlation, or a tensor/graph-based spectral residual) was found in this search.
- Microsoft's own Azure Anomaly Detector service uses **two different algorithms depending on cardinality**: SR-CNN for univariate series, and **MTAD-GAT** (Multivariate Time-series Anomaly Detection via Graph Attention Network, Zhao et al. 2020) for multivariate series — i.e., Microsoft did not extend SR to the multivariate case; they switched to an entirely different (graph-attention, not spectral) architecture — [curatedsql.com summary](https://curatedsql.com/?p=34031); MTAD-GAT original paper — [arXiv:2009.02040](https://arxiv.org/pdf/2009.02040).
- MTAD-GAT treats each univariate series as an individual feature, models cross-feature correlations explicitly via two parallel Graph Attention (GAT) layers, and temporal dependencies via a GRU — a fundamentally different (graph + recurrent) approach, not a spectral/Fourier one — [arXiv:2009.02040](https://arxiv.org/pdf/2009.02040).

### Inferences
- The absence of a joint multivariate spectral-residual method in the literature, combined with Microsoft's own choice to swap to a graph-attention model for multivariate production use, is reasonably strong indirect evidence that SR's underlying assumption (a single saliency signal from one FFT) does not transfer cleanly to capturing cross-channel anomaly patterns — the kind of anomalies (e.g., correlated drift across channels) that matter most in a 47-channel satellite telemetry setting.
- For this project's 47-channel contaminated-online setting, the only practically available "multivariate SR" is the per-channel-then-fuse pattern (as in PyOD), which is architecturally equivalent to running 47 independent univariate detectors and combining alarms — it would not capture genuinely joint/cross-channel anomaly structure, which is a known weakness relative to the already-implemented GDN (graph-based) and autoencoder-family models in this project.

### Gaps
- No controlled comparison was found of per-channel-SR-with-fusion vs. joint multivariate methods (GDN, MTAD-GAT, autoencoders) on a shared multivariate benchmark, so the magnitude of the expected performance gap for the fused-SR approach specifically could not be quantified.

---

## Is SR usable online/incrementally (sliding FFT windows) at reasonable cost for ~47 channels, 18s resolution, ~148,000 rows/month?

### Takeaway
SR is designed for streaming use via a sliding-window FFT and is deployed at large scale by Microsoft (millions of series/day), so the raw compute cost is very likely feasible on a laptop CPU for 47 channels at 18s resolution; however, no source gave a concrete big-O or measured per-point latency figure, and running it "incrementally" in the strict sense (O(1) update per new point) is not what the published method does — it recomputes a windowed FFT at every step.

### Cited Findings
- The SR method operates via a sliding window: the FFT is computed over a window of the sequence, and because the goal is low-latency detection, several estimated future points are appended after the most recent real point so that point sits near the center of the window before running SR — [Liner summary of the Microsoft paper](https://liner.com/review/timeseries-anomaly-detection-service-at-microsoft).
- Microsoft's production "Online Compute" component processes "over 4 million time series daily" using Apache Flink, i.e., SR(-CNN) is deployed at very large real-time scale in production — [arXiv:1906.03821](https://arxiv.org/pdf/1906.03821) (via search synthesis); reiterated in Microsoft Tech Community retirement-notice discussion — [curatedsql.com](https://curatedsql.com/?p=34031).
- The standalone R package `spectralAnomaly` implements SR as a general-purpose, non-training detection method for arbitrary time series — [spectralAnomaly on CRAN mirror](https://archive.linux.duke.edu/cran/web/packages/spectralAnomaly/refman/spectralAnomaly.html).

### Inferences
- A full-window FFT recomputed at every new timestep costs O(W log W) per point for window length W (standard FFT complexity); this is not independently confirmed by any source for SR specifically, but follows directly from the documented algorithm (FFT → residual → inverse FFT) being re-run on a sliding window — treat as a reasoned estimate, not a cited fact.
- For 47 channels at 18-second resolution (~4,800 points/day, ~148,000/month), even a per-channel full FFT recomputed every step (e.g., window W=100-500) is almost certainly cheap enough for a laptop CPU in an offline/batch walk-forward evaluation — Microsoft's 4-million-series/day production figure (using commodity stream processing, not specialized hardware) is suggestive that this is not computationally demanding relative to the deep-learning baselines already implemented in this project (Anomaly Transformer, autoencoders, GDN). This is an inference, not a cited benchmark number.
- True incremental/O(1)-per-point updates would require a recursive/sliding-DFT formulation rather than the published block-FFT SR method; this is a modification the original authors did not describe, so "incremental" SR as commonly implemented is really "frequent batch recomputation on a sliding window," which is still cheap at this data volume but is not asymptotically optimal.

### Gaps
- No source gave a measured wall-clock or FLOP figure for SR at any channel count/resolution, so the "reasonable cost for 47 channels" conclusion rests on inference from Microsoft's production scale claim and general FFT complexity, not a direct citation. A quick local benchmark (not found in literature) would be the only way to get a hard number.
- No source discussed a genuinely incremental (sliding-DFT) implementation of SR specifically, only the standard block-FFT version.

---

## Is SR/spectral-residual-style detection actually used in production today? What is Microsoft's own status/updates (2024-2026)?

### Takeaway
Microsoft's Azure Anomaly Detector service (built on SR-CNN for univariate and MTAD-GAT for multivariate) is being retired, with the retirement date reported as October 1, 2026 — i.e., as of today (2026-10-09) this retirement has already occurred or is imminent — and Microsoft has open-sourced the underlying algorithms as a new PyPI package rather than continuing the hosted service, while pointing users toward Microsoft Fabric's anomaly detection workflow going forward.

### Cited Findings
- Microsoft's Anomaly Detector FAQ states that as of September 20, 2023, new Anomaly Detector resources can no longer be created, and the Anomaly Detector service itself is being retired on October 1, 2026 — [Microsoft Learn Q&A: "Anomaly detector deprecated"](https://learn.microsoft.com/en-us/answers/questions/1550154/anomaly-detector-deprecated); [Anomaly Detector FAQ](https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/faq).
- A Microsoft Tech Community post states that as part of the migration process, "the anomaly detection algorithms were open sourced and published" as a new Python package, and that Microsoft now offers a time-series anomaly detection workflow inside the Microsoft Fabric data platform instead — [via search synthesis, Microsoft Tech Community](https://techcommunity.microsoft.com/tag/Anomaly%20Detector); also referenced at [curatedsql.com](https://curatedsql.com/?p=34031).
- The service's underlying algorithms are explicitly named as "SR-CNN for univariate analysis and MTAD-GAT for multivariate analysis" — confirming that in Microsoft's own most recent production architecture, spectral residual (via SR-CNN) is used only for the univariate case, never the multivariate one — [curatedsql.com](https://curatedsql.com/?p=34031).
- A 2024 Microsoft Q&A response stated there was "no new product for alternative at this moment," and a late-2023 response confirmed existing Azure Anomaly Detector/Metrics Advisor resources would continue operating until the October 2026 retirement but no new resources could be created — [Microsoft Learn Q&A](https://learn.microsoft.com/en-us/answers/questions/1434709/azure-anomaly-detection-is-being-retired-what-are).
- Microsoft's original open-source code for SR/SR-CNN lives at `github.com/microsoft/anomalydetector` (mirrored, e.g., at `github.com/emerrf/anomalydetector`), described as providing preprocessing, training, and evaluation modules, with a note that "this sample only RUN SR, for SR-CNN please refer [to a separate section]" — [GitHub mirror](https://github.com/emerrf/anomalydetector).
- An independent R package, `spectralAnomaly`, implements SR standalone, citing the 2019 Microsoft paper, with active versions reported in 2024-2025 — [spectralAnomaly on R-universe](https://al-obrien.r-universe.dev/spectralAnomaly/citation).

### Inferences
- The clearest 2024-2026 "update" on Microsoft's own use of SR is negative: they are winding down the hosted product that used it, and their own multivariate production path already stopped using spectral methods years ago in favor of graph attention (MTAD-GAT). This substantially weakens any "it's validated in production today" argument for SR as a method, especially for multivariate use — the one setting this project needs.
- The fact that Microsoft chose to retire the whole service rather than simply deprecate SR-CNN specifically does not, by itself, indicate the algorithm failed — the retirement appears to be a general product/business decision (folding capability into Fabric) rather than an algorithmic repudiation. This nuance should be stated carefully in the paper: it is a business discontinuation, not a published negative result about SR's accuracy.

### Gaps
- No official Microsoft blog/announcement text (as opposed to secondary Q&A/community summaries) was directly fetched; the retirement date and open-sourcing claim rest on secondary synthesis of search snippets rather than a primary Microsoft announcement page, which should ideally be verified by fetching the official retirement notice and Fabric migration page directly if more certainty is needed for the paper.
- The exact name and current state of the "time-series-anomaly-detector" PyPI package could not be confirmed directly (its PyPI page was not successfully fetched) — treat this package's existence/name as secondary-sourced, not primary-confirmed.

---

## Recent (2024-2026) papers applying SR, wavelets, or other spectral methods to spacecraft/satellite telemetry or comparable real industrial multivariate sensor data?

### Takeaway
No peer-reviewed 2024-2026 paper was found that applies Spectral Residual specifically — or a well-validated wavelet-based method — to spacecraft/satellite telemetry or comparable real industrial multivariate sensor data; the closest 2024-2026 satellite-telemetry work uses entirely different approaches (time-frequency attention inside a Transformer, adversarial training, Isolation Forest, or non-spectral explainable ML), none citing or benchmarking against SR.

### Cited Findings
- A 2026 Chinese Journal of Space Science paper, "Anomaly Detection for Satellite Telemetry Parameters Based on Time-Frequency Feature Analysis and Adversarial Training," adds a frequency-domain attention mechanism to a simplified Transformer combined with adversarial training — this is spectral/frequency-domain-adjacent but is not Spectral Residual, and is a Transformer-based method, not an FFT-residual method — [cjss.ac.cn](https://cjss.ac.cn/cn/article/doi/10.11728/cjss2026-0005).
- A 2024 Elsevier paper on explainable anomaly detection using two real NASA telemetry datasets incorporates "magnitude, frequency, and waveform information" into feature extraction (reporting 95.3% precision, 100% recall) but does not use Spectral Residual or wavelets — found only via search synthesis, not directly fetched.
- Simões et al. (arXiv:2408.02581, 2024) apply Isolation Forests to ESA's Ariel mission telemetry-adjacent data — spacecraft-related but not spectral at all.
- An undated (year unconfirmed) Bauman Moscow State Technical University paper discusses using the wavelet transform method for analyzing spacecraft telemetric information and proposes automating that analysis — relevant in principle but its publication year could not be confirmed as 2024-2026, and no empirical results were retrievable from the search snippet — [engjournal.bmstu.ru](https://engjournal.bmstu.ru/eng/catalog/arse/ahtp/1890.html).
- A 2026 hackathon project ("AstroAudit") uses FFT and Welch power spectral density plus spectral entropy on telemetry windows, but this is an unreviewed hackathon submission, not an academic paper — [lablab.ai](https://lablab.ai/ai-hackathons/ibm-bob-hackathon/astrophage-ai/astroaudit).
- Direct search for "spectral residual" combined with "ESA-ADB" or "ESA Anomaly Detection Benchmark" returned no results connecting the two — the ESA-ADB benchmark papers (arXiv:2406.17826, arXiv:2607.07335) do not appear to mention Spectral Residual among their compared methods (which include iForest, LOF, KNN, Telemanom, OmniAnomaly, and others per the benchmark's Table 5), though this absence was inferred from search snippets rather than confirmed by a full read of Table 5 — [arXiv:2406.17826](https://arxiv.org/pdf/2406.17826), [arXiv:2607.07335](https://arxiv.org/pdf/2607.07335).

### Inferences
- The complete absence of SR (or wavelet methods) from both the ESA-ADB benchmark and any clearly peer-reviewed 2024-2026 spacecraft telemetry paper is itself informative: the active research community working on real satellite telemetry anomaly detection has not adopted spectral-residual-style methods as a baseline or candidate, in contrast to iForest, OmniAnomaly, Telemanom, GDN-style graph methods, and deep autoencoders — all of which this project has already implemented and which do appear in the ESA-ADB comparison tables.
- This absence should be read as mild negative evidence for SR's relevance to this specific domain (nobody has found it worth testing on real multivariate spacecraft data), though it is not proof of poor performance — it may simply reflect it predating the area's benchmark culture or being seen (correctly, per the multivariate-extension findings above) as a poor architectural fit for multivariate, correlated-channel telemetry.

### Gaps
- Could not confirm the Bauman wavelet-telemetry paper's publication year, so its relevance to the 2024-2026 window is unverified.
- Did not perform a full read of ESA-ADB's Table 5 (list of all benchmarked methods) to definitively rule out SR's inclusion — this was inferred from search snippets, not a direct primary-source check. A follow-up direct read of the ESA-ADB GitHub repo's method list would close this gap with certainty.
- No access to IEEE Xplore or MDPI full-text search was performed (only generic web search), so paywalled 2024-2026 wavelet/spectral spacecraft papers may exist but were not surfaced by this search method.

---

## How does SR compare empirically against isolation-forest/autoencoder-style methods on contaminated training data specifically?

### Takeaway
No source — including the SR-specific literature, the TSB-AD benchmark, or the contaminated-training-data literature — was found that directly and empirically compares SR against IForest/autoencoder methods specifically under training-set contamination; this is a genuine, confirmed gap. Separately, the general contaminated-training-data literature establishes that residual/reconstruction-based detectors (the family autoencoders belong to) degrade under contamination unless explicitly hardened — but SR, being training-free, is architecturally exempt from "training set contamination" in the conventional sense, which is itself a notable structural difference worth flagging.

### Cited Findings
- SR requires no training step at all — it is a stateless per-window signal-processing transform applied identically to any input window, with no parameters fit to a "training set" — [Microsoft.ML documentation via source.dot.net](https://source.dot.net/Microsoft.ML.TimeSeries/SrCnnEntireAnomalyDetector.cs.html); [Liner summary](https://liner.com/review/timeseries-anomaly-detection-service-at-microsoft).
- A 2023 paper on fully-unsupervised anomaly detection with contaminated data (Ulmer, Zgraggen, Goren Huber, arXiv:2308.13352) states that residual/reconstruction-based models "typically perform poorly once anomalous samples enter the training set," motivating a data-cleaning step before training; this concerns autoencoder-style reconstruction models, not SR — [arXiv:2308.13352](https://arxiv.org/pdf/2308.13352).
- That same paper's proposed cleaning framework, evaluated against OC-SVM and Isolation Forest as simple unsupervised baselines on MIMII machine-sound data, reports that training on cleaned (vs. contaminated) data can match or exceed training on ideal anomaly-free data — but again, SR is not part of this comparison — [arXiv:2308.13352](https://arxiv.org/pdf/2308.13352) / [ZHAW repository](https://digitalcollection.zhaw.ch/handle/11475/30284).
- TSB-UAD/TSB-AD's IForest implementation derives its decision threshold from an assumed "contamination" fraction set as an input parameter — meaning IForest's own contamination-handling is itself just a tunable hyperparameter assumption, not a learned robustness property — [TSB-UAD docs](https://tsb-uad.readthedocs.io/en/latest/algorithms/Tree.html).
- No source discussing SR's behavior specifically under contaminated/noisy training data, or any paper placing SR in the same experimental protocol as the contamination-robustness literature (USDR, LOE, CRTF, etc. — the methods already implemented in this project), was found.

### Inferences
- Because SR has no training phase, "training-set contamination" (this project's core stated constraint) is not a failure mode that applies to SR the way it applies to IForest or autoencoders — SR would be evaluated purely on its per-window detection/scoring quality against whatever signal is present, with no notion of a contaminated "normal" model to poison. This is architecturally a point in SR's favor for this project's specific constraint (zero clean labels, contamination at every stage), though it does not mean SR would perform *well* — only that it sidesteps this particular failure mode.
- The lack of any direct SR vs. IForest/autoencoder contamination comparison means that any claim in the paper about SR's relative robustness to contamination would have to be framed as a structural/architectural argument (no training phase → immune to training contamination) rather than an empirically demonstrated result, since no such empirical study exists in the literature surveyed.

### Gaps
- No empirical shared-protocol benchmark comparing SR against IForest/autoencoders under training-set contamination exists in the literature found by this search. This should be flagged explicitly as an open gap in the paper rather than inferred/filled in.
- It remains untested (in the literature, and not independently verified here) whether SR's saliency-map thresholding step implicitly assumes a majority-normal distribution within each sliding window — if anomalies are dense/contiguous within a window, SR's own windowed statistics (the moving average it compares saliency against) could themselves become contaminated at inference time, which is a distinct contamination concern from "training set contamination" and was not addressed by any source found.

---

## Overall assessment for this project (synthesis, not a new cited claim)

Based on the findings above: Spectral Residual is (1) fundamentally univariate with only an ad hoc per-channel-fusion path to multivariate use, which Microsoft's own production system abandoned in favor of a graph-attention model (MTAD-GAT) for multivariate data; (2) very likely CPU-cheap enough for 47 channels at 18s resolution based on Microsoft's large-scale production deployment and standard FFT complexity, though no direct benchmark number was found; (3) structurally immune to "training-set contamination" because it has no training phase, which is a genuine point of difference from this project's already-implemented contamination-robust autoencoders — but this also means SR cannot benefit from historical normal-pattern learning the way the deep models can; (4) being phased out of Microsoft's own hosted production service (retiring ~Oct 2026) with multivariate production already migrated away from spectral methods years ago; and (5) essentially absent from the active 2024-2026 spacecraft/satellite-telemetry and ESA-ADB research literature, with no study found that tests it against this project's already-implemented baselines (IForest, autoencoders, GDN) under contamination. Taken together, SR is a legitimate, cheap, "genuinely different" method worth *mentioning* in related work as a classical frequency-domain contrast point — but the evidence does not support it as a strong empirical candidate for this project's multivariate, contaminated, online setting, and the paper should present this as a reasoned architectural judgment rather than claim an empirical multivariate/contamination result that does not exist in the literature.
