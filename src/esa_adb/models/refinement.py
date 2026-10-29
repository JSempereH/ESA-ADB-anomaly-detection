"""Label-free training-data refinement for residual anomaly detectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RefinementDiagnostics:
    input_rows: int
    retained_rows: int
    rejected_rows: int
    score_threshold: float


class USDRRefiner:
    """Approximate USDR with balanced model-membership subsets.

    A sample is suspicious when residual models that excluded it reconstruct it worse
    than models trained with it. A robust MAD cutoff decides which suspicious samples
    to exclude; this avoids supplying a contamination rate to the refiner.
    """

    def __init__(
        self,
        n_models: int = 5,
        memberships_per_row: int = 2,
        mad_multiplier: float = 3.5,
        max_reject_fraction: float = 0.05,
        seed: int = 42,
    ) -> None:
        if n_models < 2:
            raise ValueError("n_models must be at least two")
        if not 0 < memberships_per_row < n_models:
            raise ValueError("memberships_per_row must be in (0, n_models)")
        if mad_multiplier <= 0:
            raise ValueError("mad_multiplier must be positive")
        if not 0 <= max_reject_fraction < 1:
            raise ValueError("max_reject_fraction must be in [0, 1)")
        self.n_models = n_models
        self.memberships_per_row = memberships_per_row
        self.mad_multiplier = mad_multiplier
        self.max_reject_fraction = max_reject_fraction
        self.seed = seed
        self._calls = 0

    def _memberships(self, n_rows: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + self._calls)
        memberships = np.zeros((n_rows, self.n_models), dtype=bool)
        for row in range(n_rows):
            memberships[row, rng.choice(self.n_models, size=self.memberships_per_row, replace=False)] = True
        return memberships

    @staticmethod
    def scores_from_residuals(residuals: np.ndarray, memberships: np.ndarray) -> np.ndarray:
        """Compute USDR's mean excluded-minus-included normalized residual score."""
        if residuals.shape != memberships.shape:
            raise ValueError("residuals and memberships must have equal shape")
        included = memberships.sum(axis=1)
        excluded = (~memberships).sum(axis=1)
        if np.any(included == 0) or np.any(excluded == 0):
            raise ValueError("every row must be both included and excluded by a model")
        included_mean = (residuals * memberships).sum(axis=1) / included
        excluded_mean = (residuals * ~memberships).sum(axis=1) / excluded
        return excluded_mean - included_mean

    def select(
        self,
        values: np.ndarray,
        model_factory: Callable[[], Any],
        score: Callable[[Any, np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, RefinementDiagnostics]:
        """Return retained rows and diagnostics without consulting labels or future data."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < self.n_models:
            raise ValueError("values must contain at least n_models rows")

        memberships = self._memberships(values.shape[0])
        residuals = np.empty_like(memberships, dtype=np.float64)
        for model_index in range(self.n_models):
            model = model_factory()
            in_subset = memberships[:, model_index]
            model.fit(values[in_subset])
            raw_scores = np.asarray(score(model, values), dtype=np.float64)
            train_scores = raw_scores[in_subset]
            scale = train_scores.std(ddof=1) if train_scores.size > 1 else 0.0
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                scale = 1.0
            residuals[:, model_index] = (raw_scores - train_scores.mean()) / scale

        self._calls += 1
        refinement_scores = self.scores_from_residuals(residuals, memberships)
        finite_scores = refinement_scores[np.isfinite(refinement_scores)]
        if finite_scores.size == 0:
            return values, RefinementDiagnostics(len(values), len(values), 0, float("inf"))

        center = float(np.median(finite_scores))
        mad = float(np.median(np.abs(finite_scores - center)))
        threshold = center + self.mad_multiplier * 1.4826 * mad
        reject = np.isfinite(refinement_scores) & (refinement_scores > threshold)
        max_rejected = int(np.floor(self.max_reject_fraction * len(values)))
        if reject.sum() > max_rejected:
            reject = np.zeros(len(values), dtype=bool)
            if max_rejected:
                order = np.argsort(-refinement_scores, kind="stable")
                reject[order[:max_rejected]] = True

        retained = values[~reject]
        return retained, RefinementDiagnostics(
            input_rows=len(values),
            retained_rows=len(retained),
            rejected_rows=int(reject.sum()),
            score_threshold=threshold,
        )
