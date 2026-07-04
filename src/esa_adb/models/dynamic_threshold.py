from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.stats import genpareto


class RollingPercentileThreshold:
    """Model-agnostic adaptive threshold: a bounded window of recent scores believed
    nominal, thresholding at a percentile of that window -- refreshed as new scores
    arrive, instead of the single fixed threshold every detector in this project
    otherwise calibrates once (`GlobalSTD.n_std`, `LSTMForecaster.threshold_`,
    `PCAReconstructor.threshold_`).

    Directly targets a blind spot a fixed threshold has: the paper documents concept
    drift late in Mission1 (channel groups 4, 7, 13, last 18 months), where a channel's
    "normal" range genuinely shifts over the mission's lifetime. A fixed threshold
    calibrated early on either stays too tight (false positives pile up as the nominal
    distribution drifts) or too loose (real anomalies start slipping through) -- a
    rolling threshold tracks the drift instead of assuming it away. This wraps *any*
    detector's `score()` output; it doesn't replace or know about the detector itself.

    There is an unavoidable trade-off in how the window gets updated, controlled by
    `protect_from_contamination`:
    - `False` (default): every score is folded into the window, flagged or not. This is
      what actually lets the threshold track genuine gradual drift -- but it means a
      *sustained* anomaly episode will, after `window_size` samples, get absorbed into
      the "nominal" window and stop being flagged even though it's still ongoing.
    - `True`: only scores that weren't flagged get folded in, so a sustained anomaly
      can't drag the threshold up to match it -- but as the direct consequence, genuine
      drift that gets flagged can *never* be caught back up to, since it's permanently
      excluded from the window that would otherwise adapt to it.
    There's no way to tell "the world changed" from "this is an ongoing anomaly" from the
    score stream alone -- the same fundamental ambiguity the paper calls out for rare
    nominal events (R7). Pick based on which failure mode matters more for your case.

    `recompute_every` controls how often the threshold is actually recalculated (`1` =
    every single update, the exact/default behaviour). `threshold` itself is cached and
    O(1) to read -- `np.percentile` sorts the whole window, so recomputing it on every
    sample of a long stream is not just unnecessary but genuinely catastrophic at the
    scale of a full mission's row count (millions of rows). A production monitor has no
    need to re-sort a 20,000-sample window on every single new 18-second tick;
    recomputing every few hundred/thousand samples is a negligible drift-tracking cost.
    """

    def __init__(
        self,
        window_size: int = 5000,
        percentile: float = 99.0,
        initial_scores: Optional[np.ndarray] = None,
        protect_from_contamination: bool = False,
        recompute_every: int = 1,
    ):
        self.window_size = window_size
        self.percentile = percentile
        self.protect_from_contamination = protect_from_contamination
        self.recompute_every = recompute_every
        self._buffer: deque = deque(maxlen=window_size)
        self._cached_threshold = float("inf")
        self._since_recompute = 0
        if initial_scores is not None:
            self._buffer.extend(float(s) for s in initial_scores)
        self._recompute()

    def _recompute(self) -> None:
        self._cached_threshold = float(np.percentile(self._buffer, self.percentile)) if self._buffer else float("inf")
        self._since_recompute = 0

    @property
    def threshold(self) -> float:
        return self._cached_threshold

    def update(self, score: float) -> None:
        self._buffer.append(float(score))
        self._since_recompute += 1
        if self._since_recompute >= self.recompute_every:
            self._recompute()

    def predict_and_update(self, score: float) -> bool:
        is_anomalous = score > self.threshold
        if is_anomalous and self.protect_from_contamination:
            return is_anomalous
        self.update(score)
        return is_anomalous

    def predict_stream(self, scores: np.ndarray) -> np.ndarray:
        """Process a whole array of scores in chronological order, one at a time."""
        preds = np.zeros(len(scores), dtype=bool)
        for i, s in enumerate(scores):
            preds[i] = self.predict_and_update(s)
        return preds

    def predict_batch(self, scores: np.ndarray) -> np.ndarray:
        """Vectorized equivalent of `predict_stream`: processes `scores` in chunks that
        never cross a recompute boundary, instead of one Python-level method call per
        sample. `recompute_every` already amortizes `np.percentile`'s own cost (see this
        class's docstring), but at this project's real walk-forward scale (millions of
        rows per fold) the per-sample Python loop itself becomes the next bottleneck.

        Bit-identical to `predict_stream` when `protect_from_contamination=False` (this
        project's default): each chunk is predicted using the threshold cached at the
        chunk's start, then folded into the buffer as one batch, mirroring exactly how
        many raw samples the per-sample loop would have consumed before its next
        recompute. Under `protect_from_contamination=True`, a chunk may contain both
        flagged and unflagged samples -- those are still predicted with the same
        pre-chunk threshold and only the unflagged ones folded in, matching per-sample
        behavior *within* a chunk, but the recompute cadence itself is based on raw
        sample count rather than accepted-update count, so it can drift slightly from
        `predict_stream`'s exact schedule in that one mode.
        """
        scores = np.asarray(scores, dtype=float)
        n = scores.shape[0]
        preds = np.zeros(n, dtype=bool)
        pos = 0
        while pos < n:
            take = max(1, min(self.recompute_every - self._since_recompute, n - pos))
            chunk = scores[pos : pos + take]
            flags = chunk > self.threshold
            preds[pos : pos + take] = flags
            to_fold_in = chunk[~flags] if self.protect_from_contamination else chunk
            self._buffer.extend(to_fold_in.tolist())
            self._since_recompute += take
            if self._since_recompute >= self.recompute_every:
                self._recompute()
            pos += take
        return preds


class POTThreshold:
    """Streaming Peaks-Over-Threshold / Extreme Value Theory dynamic threshold --
    Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory* (KDD 2017),
    the thresholding mechanism several well-known deep anomaly detectors (OmniAnomaly,
    USAD, ...) use in place of an arbitrary fixed percentile like
    `RollingPercentileThreshold`'s.

    `RollingPercentileThreshold` answers "what's the Nth percentile of recent scores" --
    a reasonable rule of thumb, but the percentile itself (99.0 throughout this project)
    is picked by hand and has no relationship to how heavy-tailed the score distribution
    actually is. POT instead: (1) picks a moderate "peak" level `t` (the
    `init_threshold_percentile`, e.g. 98th) from an initial batch, (2) fits a
    Generalized Pareto Distribution to the *excesses* over `t` (the values that make it
    into the tail at all) via MLE, and (3) extrapolates from that fitted tail to the
    quantile corresponding to a target false-alarm probability `risk` (e.g. 1e-3) --
    i.e. the actual anomaly threshold is derived from the shape of the tail itself,
    not assumed to sit at a fixed percentile of it.

    Streaming update: each new score below `t` only grows `n` (the total count, used to
    rescale the extreme-quantile estimate); each score above `t` also grows the excess
    set, and every `recompute_every` updates the GPD is refit and the threshold
    re-derived -- same amortization concern as `RollingPercentileThreshold`, but sharper
    here: `genpareto.fit`'s MLE optimization costs on the order of 10-20ms per call on a
    ~1000-1500-point excess set, so refitting on every single sample would be
    catastrophic over millions of rows, and even the default `recompute_every=200` adds
    up fast across many folds/models in a walk-forward comparison -- pick a larger value,
    e.g. 1000-2000, when wrapping a long real stream rather than a short unit test. `t`
    itself, unlike
    `RollingPercentileThreshold`'s whole window, is set once from the initial batch and
    never recomputed -- matching the original SPOT algorithm's design (the peak level is
    meant to be a stable, moderate reference point; only the extrapolated extreme
    quantile above it should move as more tail data accumulates).

    Falls back to a plain `RollingPercentileThreshold` (same initial batch) whenever
    there aren't yet `min_excesses` points above `t` to fit a GPD from, or the MLE fit
    fails/produces a degenerate (non-finite, sub-peak) quantile -- a GPD fit on a
    handful of points is unstable enough that trusting it uncritically would be worse
    than the plain percentile it's trying to improve on.
    """

    def __init__(
        self,
        risk: float = 1e-3,
        init_threshold_percentile: float = 98.0,
        window_size: int = 20_000,
        min_excesses: int = 20,
        initial_scores: Optional[np.ndarray] = None,
        recompute_every: int = 200,
    ):
        self.risk = risk
        self.init_threshold_percentile = init_threshold_percentile
        self.window_size = window_size
        self.min_excesses = min_excesses
        self.recompute_every = recompute_every
        self._excesses: deque = deque(maxlen=window_size)
        self._t: Optional[float] = None  # the fixed "peak" threshold excesses are measured above
        self._n = 0  # total observations seen since init
        self._since_recompute = 0
        self._cached_threshold = float("inf")
        self._fallback = RollingPercentileThreshold(window_size=window_size, percentile=99.0, recompute_every=recompute_every)
        if initial_scores is not None and len(initial_scores) > 0:
            self._init_from(np.asarray(initial_scores, dtype=float))
        self._recompute()

    def _init_from(self, scores: np.ndarray) -> None:
        self._t = float(np.percentile(scores, self.init_threshold_percentile))
        self._n = scores.shape[0]
        self._excesses.extend((scores[scores > self._t] - self._t).tolist())
        self._fallback = RollingPercentileThreshold(
            window_size=self.window_size, percentile=99.0, initial_scores=scores, recompute_every=self.recompute_every
        )

    def _recompute(self) -> None:
        self._since_recompute = 0
        if self._t is None or len(self._excesses) < self.min_excesses:
            self._cached_threshold = self._fallback.threshold
            return
        excesses = np.array(self._excesses)
        try:
            shape, _, scale = genpareto.fit(excesses, floc=0)
        except Exception:
            self._cached_threshold = self._fallback.threshold
            return
        n_t = len(excesses)
        q = self.risk
        if abs(shape) < 1e-6:  # exponential tail, the xi -> 0 limit of the GPD quantile formula
            z_q = self._t - scale * np.log(q * self._n / n_t)
        else:
            z_q = self._t + (scale / shape) * ((q * self._n / n_t) ** (-shape) - 1)
        # An extreme quantile can never legitimately sit below the peak level it was
        # extrapolated from -- a fit that implies otherwise is degenerate, not a
        # genuinely tighter threshold.
        if not np.isfinite(z_q) or z_q < self._t:
            self._cached_threshold = self._fallback.threshold
        else:
            self._cached_threshold = float(z_q)

    @property
    def threshold(self) -> float:
        return self._cached_threshold

    def update(self, score: float) -> None:
        self._n += 1
        self._fallback.update(score)
        if self._t is not None and score > self._t:
            self._excesses.append(score - self._t)
        self._since_recompute += 1
        if self._since_recompute >= self.recompute_every:
            self._recompute()

    def predict_and_update(self, score: float) -> bool:
        is_anomalous = score > self.threshold
        self.update(score)
        return is_anomalous

    def predict_stream(self, scores: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(scores), dtype=bool)
        for i, s in enumerate(scores):
            preds[i] = self.predict_and_update(s)
        return preds

    def predict_batch(self, scores: np.ndarray) -> np.ndarray:
        """Vectorized equivalent of `predict_stream`, kept for API symmetry with
        `RollingPercentileThreshold.predict_batch` and because it's still a real (if
        modest) win on its own prediction/buffer-update overhead -- but it does NOT
        address this class's dominant cost: each `genpareto.fit` MLE call runs exactly as
        many times either way, since both methods share the same `recompute_every`
        cadence. `recompute_every` itself is the actual lever on total runtime for this
        class -- see `test_pot_recompute_every_is_the_real_lever_on_runtime`.
        Bit-identical to `predict_stream`: unlike `RollingPercentileThreshold`, there's no
        `protect_from_contamination`-style branching here to approximate -- every sample
        always counts toward `n` and the excess set the same way regardless of whether it
        gets flagged.
        """
        scores = np.asarray(scores, dtype=float)
        n = scores.shape[0]
        preds = np.zeros(n, dtype=bool)
        pos = 0
        while pos < n:
            take = max(1, min(self.recompute_every - self._since_recompute, n - pos))
            chunk = scores[pos : pos + take]
            preds[pos : pos + take] = chunk > self.threshold
            self._n += chunk.shape[0]
            self._fallback.predict_batch(chunk)
            if self._t is not None:
                self._excesses.extend((chunk[chunk > self._t] - self._t).tolist())
            self._since_recompute += take
            if self._since_recompute >= self.recompute_every:
                self._recompute()
            pos += take
        return preds
