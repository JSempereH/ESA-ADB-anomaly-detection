from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.lightweight.pca_reconstruction import PCAReconstructor


class SubsystemPCAEnsemble:
    """One `PCAReconstructor` per subsystem group of channels, instead of a single joint
    model across every target channel -- motivated by `channels.csv`'s own
    `subsystem`/`group` metadata and by `PCAReconstructor`'s confirmed real-data
    weakness: 47 heterogeneous channels bundled into one PCA fit can dilute genuinely
    correlated structure *within* a subsystem with unrelated noise from every other one.

    A row is flagged if **any** subsystem's own detector flags it by default (OR
    combination, `min_agreeing=1`) -- subsystems are largely independent physical parts
    of the spacecraft, so an anomaly local to one shouldn't need corroboration from
    unrelated subsystems to be reported in principle (contrast with
    `PairedLearnerEnsemble`'s AND combination, which combines two views of the *same*
    channels rather than independent subsystems).

    In practice, real Mission2 data showed this OR rule is a *bad* default: independent
    per-subsystem false-positive rates compound instead of cancelling out (79,843
    detections vs. 11,069 for one joint `PCAReconstructor` over the same test period --
    each subsystem is small enough, and therefore noisy enough, that almost every row
    trips at least one of them). `predict(..., min_agreeing=k)` requires k-or-more
    subsystems to agree before flagging a row, trading recall (an anomaly truly confined
    to one subsystem is missed unless k=1) for a sharp drop in compounded false
    positives -- the model-agnostic analogue of `PairedLearnerEnsemble`'s AND rule,
    generalized from 2 detectors to N.
    """

    def __init__(self, channel_groups: dict[str, list[int]], min_channels: int = 3, n_components: float = 0.95):
        """`channel_groups`: {subsystem_name: [column indices into the target-channel
        value array]} -- built by the caller from `channels.csv`, since this class
        doesn't know about the raw metadata format. Groups smaller than `min_channels`
        are dropped (too few channels for PCA to find meaningful cross-channel structure).
        """
        self.channel_groups = {name: idx for name, idx in channel_groups.items() if len(idx) >= min_channels}
        self.n_components = n_components
        self.models: dict[str, PCAReconstructor] = {}

    def fit(self, X: np.ndarray) -> "SubsystemPCAEnsemble":
        for name, idx in self.channel_groups.items():
            self.models[name] = PCAReconstructor(n_components=self.n_components).fit(X[:, idx])
        return self

    def partial_fit(self, X: np.ndarray) -> "SubsystemPCAEnsemble":
        """Forward to each subsystem's own `PCAReconstructor.partial_fit` -- creates a
        subsystem's model on first use (mirroring `PCAReconstructor.partial_fit`'s own
        self-initializing first call) rather than requiring a prior `fit()`.
        """
        for name, idx in self.channel_groups.items():
            if name not in self.models:
                self.models[name] = PCAReconstructor(n_components=self.n_components)
            self.models[name].partial_fit(X[:, idx])
        return self

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "SubsystemPCAEnsemble":
        for name, idx in self.channel_groups.items():
            self.models[name].calibrate_threshold(validation_values[:, idx], percentile=percentile)
        return self

    def score_per_subsystem(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {name: self.models[name].score(X[:, idx]) for name, idx in self.channel_groups.items()}

    def predict(self, X: np.ndarray, thresholds: Optional[dict[str, float]] = None, min_agreeing: int = 1) -> np.ndarray:
        """`min_agreeing=1` (default) is the original OR rule: any single subsystem's
        detector is enough. `min_agreeing>1` requires that many *independent* subsystems
        to flag the same row before it's reported -- see the class docstring for why the
        default is a confirmed bad choice on real data.
        """
        n = X.shape[0]
        agreement = np.zeros(n, dtype=np.int32)
        for name, idx in self.channel_groups.items():
            thr = thresholds.get(name) if thresholds else None
            agreement += self.models[name].predict(X[:, idx], thr)
        return agreement >= min_agreeing
