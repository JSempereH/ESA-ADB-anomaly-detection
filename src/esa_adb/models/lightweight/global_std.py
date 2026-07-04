from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from esa_adb.models.base import AnomalyDetector


class GlobalSTD(AnomalyDetector):
    """Per-channel k-sigma detector: the paper's GlobalSTD baseline.

    Any sample deviating from a channel's train-nominal mean by more than `n_std`
    standard deviations is anomalous. Cannot detect anomalies that only show up as
    cross-channel correlation breaks, and is blind to concept drift once fit -- but it's
    the simplest possible interpretable baseline, essentially free to compute, and is
    what satellite operators already do manually via fixed nominal limits.
    """

    def __init__(self, n_std: float = 3.0):
        self.n_std = n_std
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "GlobalSTD":
        self.mean_ = X.mean(axis=0)
        # ddof=1 (sample std) to match OnlineGlobalSTD's Welford estimate exactly.
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # constant channels: avoid divide-by-zero, matches preprocessing/standardize.py
        self.std_ = std
        return self

    @classmethod
    def from_standardization_stats(
        cls, stats: pl.DataFrame, channels: list[str], n_std: float = 3.0
    ) -> "GlobalSTD":
        """Build directly from the pipeline's precomputed `standardization_stats.parquet`
        instead of refitting -- those per-channel train-nominal mean/std are already
        exactly what GlobalSTD needs.
        """
        by_channel = {row["channel"]: row for row in stats.iter_rows(named=True)}
        missing = [c for c in channels if c not in by_channel]
        if missing:
            raise ValueError(f"Channels missing from standardization_stats: {missing[:5]}...")
        obj = cls(n_std=n_std)
        obj.mean_ = np.array([by_channel[c]["mean"] for c in channels])
        obj.std_ = np.array([by_channel[c]["std"] for c in channels])
        return obj

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        """`(n_rows, n_channels)` absolute z-score per channel -- the explainability
        primitive `score()` maxes over. Since every channel is thresholded
        independently here (unlike `PCAReconstructor`/`Autoencoder`), this is already
        exactly the per-channel anomaly signal, not an approximation of one.
        """
        if self.mean_ is None:
            raise RuntimeError("GlobalSTD must be fit before scoring")
        return np.abs((X - self.mean_) / self.std_)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.score_per_channel(X).max(axis=1)  # worst-offending channel drives the row-level score

    def most_anomalous_channels(self, X: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        """For each row, the `top_n` channels with the highest z-score -- which channel(s)
        an operator should look at first for a flagged row.
        """
        z = self.score_per_channel(X)
        top_n = min(top_n, z.shape[1])
        order = np.argsort(-z, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        return self.score(X) > (threshold if threshold is not None else self.n_std)


class OnlineGlobalSTD(AnomalyDetector):
    """Streaming GlobalSTD: mean/std updated one sample at a time via Welford's algorithm.

    Unlike `GlobalSTD`, this never needs a batch of historical nominal data to fit --
    it starts from nothing and refines its notion of "nominal" as real telemetry arrives,
    which is the actual constraint on day 1 of a newly launched satellite (no representative
    history exists yet). `partial_fit()` should only be called with samples believed nominal
    (e.g. ones a detector didn't flag), so the estimate doesn't drift toward anomalous data.
    """

    def __init__(self, n_channels: int, n_std: float = 3.0):
        self.n_std = n_std
        self.n = 0
        self.mean_ = np.zeros(n_channels)
        self._m2 = np.zeros(n_channels)

    def partial_fit(self, x: np.ndarray) -> "OnlineGlobalSTD":
        self.n += 1
        delta = x - self.mean_
        self.mean_ += delta / self.n
        self._m2 += delta * (x - self.mean_)
        return self

    def fit(self, X: np.ndarray) -> "OnlineGlobalSTD":
        for x in X:
            self.partial_fit(x)
        return self

    @property
    def std_(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean_)
        std = np.sqrt(self._m2 / (self.n - 1))
        std[std == 0] = 1.0
        return std

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        """See `GlobalSTD.score_per_channel` -- identical semantics, streaming stats."""
        return np.abs((X - self.mean_) / self.std_)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.score_per_channel(X).max(axis=1)

    def most_anomalous_channels(self, X: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        """See `GlobalSTD.most_anomalous_channels`."""
        z = self.score_per_channel(X)
        top_n = min(top_n, z.shape[1])
        order = np.argsort(-z, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        return self.score(X) > (threshold if threshold is not None else self.n_std)
