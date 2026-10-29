"""Student-t version of GaussianCopulaDetector.

Satellite sensor spikes can be more extreme than a Gaussian model expects. A
t-copula handles that: it scores a deviation on a log scale instead of a quadratic
one, so one large spike doesn't dominate the score if the rest of the row still
looks like a normal joint pattern.

It shares GaussianCopulaDetector's empirical-CDF marginals and rank correlation
estimate (rank correlation doesn't change under this kind of transform, so the two
detectors end up with the same correlation matrix). Only the final marginal and
scoring formula differ: quantiles come from a Student-t distribution with a fixed
number of degrees of freedom, and the row score uses the multivariate-t log
density instead of a plain Mahalanobis distance.

The degrees of freedom are fixed, not fit from data. That keeps the model
closed-form, at the cost of not calibrating exactly how heavy the tails should be.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from esa_adb.models.lightweight.gaussian_copula import GaussianCopulaDetector


class TCopulaDetector(GaussianCopulaDetector):
    def __init__(self, df: float = 6.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.df = df

    def _pit_z(self, X: np.ndarray) -> np.ndarray:
        # Only the marginal changes here. _pit_u (the rank transform) is inherited
        # unchanged, so fit() ends up with the same filtered rows and correlation
        # matrix as the base class.
        return student_t.ppf(self._pit_u(X), df=self.df)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("TCopulaDetector must be fit before scoring")
        z = self._pit_z(np.asarray(X, dtype=np.float64))
        d2 = np.einsum("ti,ij,tj->t", z, self.corr_inv_, z)
        return (self.df + self.d_) * np.log1p(d2 / self.df)
