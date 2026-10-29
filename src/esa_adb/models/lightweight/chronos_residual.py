"""Zero-shot foundation-model forecaster (Chronos-Bolt, Ansari et al. 2024) scored by the
joint Mahalanobis distance of its forecast residuals.

The only candidate here that needs NO mission history to forecast: the weights are
pretrained, so on day 1 of a new spacecraft the forecaster is already usable and only
the small residual covariance has to be estimated (from a subsample of the first batch,
so a few thousand rows suffice). Everything else in the catalog has to learn the
telemetry first.

Cost control: the model forecasts `horizon` rows per call from the previous `context`
rows, per channel, with a stride of `horizon`, so every row is scored from a forecast
made before it was seen and a month costs n_rows/horizon x n_channels series, not
n_rows x n_channels. Rows before the first full context of a chunk score 0.

Requires `uv pip install chronos-forecasting` (deliberately not a project dependency).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.lightweight.residual_mahalanobis import ResidualMahalanobis

_PIPELINES: dict[str, object] = {}


def _pipeline(model_id: str):
    if model_id not in _PIPELINES:
        import torch
        from chronos import BaseChronosPipeline

        _PIPELINES[model_id] = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    return _PIPELINES[model_id]


class ChronosResidualDetector(AnomalyDetector):
    def __init__(self, model_id: str = "amazon/chronos-bolt-tiny", context: int = 128, horizon: int = 64,
                 fit_blocks: int = 150, batch_size: int = 1024, threads: int = 2, seed: int = 42):
        self.model_id, self.context, self.horizon = model_id, context, horizon
        self.fit_blocks, self.batch_size, self.threads, self.seed = fit_blocks, batch_size, threads, seed
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.maha = ResidualMahalanobis()

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(X, dtype=np.float64) - self.mean_) / self.std_, -10.0, 10.0)

    def _residual_blocks(self, Xs: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """(len(starts), horizon, d): actual minus the median forecast made from the
        `context` rows before each start."""
        import torch

        pipe = _pipeline(self.model_id)
        n_blocks, d = len(starts), Xs.shape[1]
        ctx = np.stack([Xs[s - self.context : s] for s in starts])  # (B, C, d)
        series = torch.tensor(ctx.transpose(0, 2, 1).reshape(n_blocks * d, self.context), dtype=torch.float32)
        preds = []
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            with torch.inference_mode():
                for i in range(0, series.shape[0], self.batch_size):
                    _, mean = pipe.predict_quantiles(series[i : i + self.batch_size], prediction_length=self.horizon, quantile_levels=[0.5])
                    preds.append(mean.numpy())
        finally:
            torch.set_num_threads(prev_threads)
        pred = np.concatenate(preds).reshape(n_blocks, d, self.horizon).transpose(0, 2, 1)
        actual = np.stack([Xs[s : s + self.horizon] for s in starts])
        return actual - pred

    def _starts(self, n_rows: int) -> np.ndarray:
        return np.arange(self.context, n_rows - self.horizon + 1, self.horizon)

    def partial_fit(self, X: np.ndarray) -> "ChronosResidualDetector":
        X = np.asarray(X, dtype=np.float64)
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            self.std_ = std
        starts = self._starts(X.shape[0])
        if len(starts) == 0:
            return self
        rng = np.random.default_rng(self.seed)
        pick = np.sort(rng.choice(starts, size=min(self.fit_blocks, len(starts)), replace=False))
        res = self._residual_blocks(self._standardize(X), pick)
        self.maha.update(res.reshape(-1, res.shape[-1]))
        return self

    def fit(self, X: np.ndarray) -> "ChronosResidualDetector":
        self.mean_ = None
        self.maha = ResidualMahalanobis()
        return self.partial_fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.maha.precision_ is None:
            raise RuntimeError("ChronosResidualDetector must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        starts = self._starts(X.shape[0])
        if len(starts) == 0:
            return out
        res = self._residual_blocks(self._standardize(X), starts)
        scores = self.maha.score(res.reshape(-1, res.shape[-1])).reshape(len(starts), self.horizon)
        for s, row in zip(starts, scores):
            out[s : s + self.horizon] = row
        return out
