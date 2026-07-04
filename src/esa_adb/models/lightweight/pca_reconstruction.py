from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from esa_adb.models.base import AnomalyDetector


class PCAReconstructor(AnomalyDetector):
    """Reconstruction-error anomaly detector via truncated PCA (equivalently, truncated
    SVD) -- the paper's own "Principal Components Classifier" (PCC) unsupervised
    baseline. Also the mathematically exact linear special case of an autoencoder: for
    purely linear reconstruction, a linear autoencoder has no advantage over truncated
    SVD, so "rank-reduction autoencoder" / "autoencoder with SVD" collapse to this same
    method rather than needing a separate deep model.

    Unlike `GlobalSTD`, which thresholds every channel independently, PCA reconstruction
    error is a genuinely *joint* signal across channels: an anomaly that keeps each
    individual channel within its own nominal range but breaks the *correlation
    structure* between channels (exactly what makes `GlobalSTD` flag ~95% of everything
    on real Mission2 data -- it can't see correlation at all) still produces a high
    reconstruction error here, because the low-rank subspace fit on nominal data no
    longer explains it well.

    Channels are z-scored before fitting PCA (their raw physical units/scales differ
    wildly -- without this, PCA would be dominated by whichever channel happens to have
    the largest raw variance rather than genuine signal structure).
    """

    def __init__(self, n_components: Union[float, int] = 0.95):
        self.n_components = n_components
        # sklearn's PCA can pick a randomized SVD solver for large data depending on the
        # requested number of components -- fix random_state so this is reproducible
        # regardless of solver choice, even though a fixed n_components as a variance
        # ratio (e.g. 0.95, this project's default) currently forces the exact 'full'
        # solver in practice.
        self.pca = PCA(n_components=n_components, random_state=42)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "PCAReconstructor":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self.pca.fit(self._standardize(X))
        return self

    def partial_fit(self, X: np.ndarray) -> "PCAReconstructor":
        """Incrementally update the subspace via sklearn's `IncrementalPCA`, without
        re-seeing any previous batch's raw rows -- the genuine "update from the new
        month, don't forget the rest" counterpart to `fit()`'s one-shot batch PCA.

        `mean_`/`std_` (the standardization stats) are established once, from the FIRST
        batch this method ever sees, and never recomputed afterwards -- recomputing them
        from just the newest batch would silently redefine "normal" every call, exactly
        the history this method exists to keep.

        `IncrementalPCA` needs a fixed integer `n_components` (unlike `PCA`, which also
        accepts a variance-ratio float like this class's own 0.95 default): the first
        call resolves that integer from a one-off variance-ratio `PCA` fit on this same
        batch, then commits to it for every later call, replacing `self.pca` with an
        `IncrementalPCA` for good (a plain `fit()` call afterwards would just overwrite
        it again with a fresh batch `PCA`, so don't mix `fit()` and `partial_fit()` on
        the same instance once streaming has started).
        """
        X = np.asarray(X, dtype=np.float64)
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            self.std_ = std
        X_std = self._standardize(X)
        if not isinstance(self.pca, IncrementalPCA):
            n_components = self.n_components
            if not isinstance(n_components, int):
                n_components = int(PCA(n_components=n_components, random_state=42).fit(X_std).n_components_)
            n_components = max(1, min(n_components, X_std.shape[0], X_std.shape[1]))
            self.pca = IncrementalPCA(n_components=n_components)
        self.pca.partial_fit(X_std)
        return self

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "PCAReconstructor":
        """Set `threshold_` from a percentile of *held-out nominal* reconstruction
        error, mirroring `LSTMForecaster`'s calibration -- using the training set itself
        for this would underestimate real error the same way it did there.
        """
        self.threshold_ = float(np.percentile(self.score(validation_values), percentile))
        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def score_per_channel(self, X: np.ndarray) -> np.ndarray:
        """`(n_rows, n_channels)` squared reconstruction error per channel, in
        standardized units -- the explainability primitive `score()`'s row-level RMSE
        averages over. A channel with a large entry here is one the fitted low-rank
        subspace failed to reconstruct for that row, i.e. a likely contributor to why
        the row was flagged (unlike `GlobalSTD`, no single channel needs to be
        individually out-of-range for this to be large -- it's the *joint* fit that broke).
        """
        if self.mean_ is None:
            raise RuntimeError("PCAReconstructor must be fit before scoring")
        X_std = self._standardize(X)
        X_recon = self.pca.inverse_transform(self.pca.transform(X_std))
        return (X_std - X_recon) ** 2

    def most_anomalous_channels(self, X: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        """For each row, the `top_n` channels with the highest reconstruction error."""
        err = self.score_per_channel(X)
        top_n = min(top_n, err.shape[1])
        order = np.argsort(-err, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(self.score_per_channel(X).mean(axis=1))

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise RuntimeError("No threshold: pass one explicitly or call calibrate_threshold() first")
        return self.score(X) > thr
