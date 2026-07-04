from __future__ import annotations

from typing import Optional

import numpy as np


class PairedLearnerEnsemble:
    """The anomaly-scoring analogue of Bach & Maloof's *Paired Learners for Concept
    Drift* (IEEE, 2008): a "stable" detector trained on all history seen so far, paired
    with a "reactive" detector trained only on a recent window. The original technique
    uses the *disagreement* between the two to detect drift and decide when to reset the
    stable learner; here, disagreement is used directly as the anomaly decision instead.

    Flags a point as anomalous only when **both** detectors flag it: if a genuine
    concept drift occurs, the reactive detector (short memory) adapts to the new normal
    and stops flagging it, while the stable detector (long memory) keeps flagging it
    since it still expects the old normal -- that "only stable flags" case is evidence
    of settled drift, not an ongoing anomaly, so it's deliberately *not* alarmed on here.
    Only when the reactive detector -- which has had the least time to normalize
    anything -- also considers a point unusual does this count as a real anomaly. This
    directly targets precision, which is what F-beta with beta<1 (this project's F0.5)
    weights more heavily than recall.

    Both underlying detectors must already implement `predict(X, threshold=...)`
    (this project's `AnomalyDetector` interface); this class doesn't fit or know
    anything about their internals, only combines their predictions.
    """

    def __init__(self, stable, reactive):
        self.stable = stable
        self.reactive = reactive

    def predict(
        self,
        X: np.ndarray,
        stable_threshold: Optional[float] = None,
        reactive_threshold: Optional[float] = None,
    ) -> np.ndarray:
        stable_pred = self.stable.predict(X, stable_threshold) if stable_threshold is not None else self.stable.predict(X)
        reactive_pred = self.reactive.predict(X, reactive_threshold) if reactive_threshold is not None else self.reactive.predict(X)
        return stable_pred & reactive_pred

    def agreement_stats(
        self,
        X: np.ndarray,
        stable_threshold: Optional[float] = None,
        reactive_threshold: Optional[float] = None,
    ) -> dict:
        """Breakdown of how often the two detectors agree/disagree -- useful to check
        the pairing is actually doing something (if `only_stable` is ~0, the reactive
        detector is barely more permissive than the stable one and the pairing adds
        little; if `both_flag` is ~0, the AND combination may be too conservative).
        """
        stable_pred = self.stable.predict(X, stable_threshold) if stable_threshold is not None else self.stable.predict(X)
        reactive_pred = self.reactive.predict(X, reactive_threshold) if reactive_threshold is not None else self.reactive.predict(X)
        return {
            "both_flag": int((stable_pred & reactive_pred).sum()),
            "only_stable": int((stable_pred & ~reactive_pred).sum()),
            "only_reactive": int((~stable_pred & reactive_pred).sum()),
            "neither": int((~stable_pred & ~reactive_pred).sum()),
        }
