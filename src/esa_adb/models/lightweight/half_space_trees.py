from __future__ import annotations

from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector


class HalfSpaceTrees(AnomalyDetector):
    """Online/incremental analogue of `WindowedIsolationForest` -- Tan, Ting & Liu's
    Half-Space Trees (*Fast Anomaly Detection for Streaming Data*, IJCAI 2011).

    `WindowedIsolationForest` has no `partial_fit`: incorporating a new month of nominal
    data means throwing away the old model and refitting from scratch on the union (or
    training a brand new one for the walk-forward "sliding" variant), an O(all data seen)
    cost repeated every fold. Half-Space Trees splits fitting into two phases instead:

    1. **Structure** (`fit()`, once): each of `n_trees` trees recursively halves a
       per-channel working range (estimated from the first batch, then padded by a random
       margin so the range isn't overfit to exactly the first batch's extremes) into a
       *complete* binary tree of `max_depth` levels -- split points are random and fixed
       at construction, never revisited, unlike Isolation Forest's data-driven splits.
    2. **Mass profile** (`partial_fit()`, incremental): each row increments a
       `latest_mass` counter at the leaf it falls into. Every `window_size` rows,
       `latest_mass` becomes the `ref_mass` used for scoring and resets to zero -- a
       lightweight two-window swap standing in for retraining, so a new nominal batch is
       absorbed with a single streaming pass instead of a batch refit.

    Score for a row is `sum_trees ref_mass(leaf)`: a leaf with few nominal points means
    that region of the working space is sparsely populated, i.e. the row was routed to an
    unusual region -- same intuition as Isolation Forest's path length, from a counted
    mass profile instead of a freshly rebuilt tree. (The original paper additionally
    weights each leaf's mass by `2^depth`; every leaf here sits at the same fixed
    `max_depth` since the tree is never pruned, so that weight is a constant factor and
    is dropped -- it would rescale every score identically without changing rankings or
    thresholds.)

    Every tree has an identical, fixed depth, so leaf routing is vectorized across all
    rows and trees at once (`_leaf_indices`) rather than walked node-by-node in Python --
    this is what makes `partial_fit`/`score` tractable on millions of rows, unlike a
    naive per-row tree-object traversal.

    Operates on raw per-row channel vectors (unlike `WindowedIsolationForest`, which
    needs a fixed-size feature vector per window) -- one streaming instance is one
    timestep's channel readings, matching the paper's own streaming formulation.
    """

    def __init__(
        self,
        n_channels: int,
        n_trees: int = 25,
        max_depth: int = 15,
        window_size: int = 250,
        size_limit_multiplier: float = 1.0,
        seed: int = 42,
    ):
        self.n_channels = n_channels
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.window_size = window_size
        self.size_limit_multiplier = size_limit_multiplier
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.n_internal = 2**max_depth - 1
        self.n_leaves = 2**max_depth
        self.feature: Optional[np.ndarray] = None  # (n_trees, n_internal) int64
        self.split: Optional[np.ndarray] = None  # (n_trees, n_internal) float64
        self.ref_mass: Optional[np.ndarray] = None  # (n_trees, n_leaves) int64
        self.latest_mass: Optional[np.ndarray] = None  # (n_trees, n_leaves) int64
        self.threshold_: Optional[float] = None
        self._seen_in_window = 0

    def _build_trees(self, bounds: np.ndarray) -> None:
        n_channels = bounds.shape[0]
        feature = np.zeros((self.n_trees, self.n_internal), dtype=np.int64)
        split = np.zeros((self.n_trees, self.n_internal), dtype=np.float64)
        for t in range(self.n_trees):
            pending = {0: bounds.copy()}
            for node_id in range(self.n_internal):
                b = pending.pop(node_id)
                f = int(self._rng.integers(n_channels))
                low, high = b[f]
                s = (low + high) / 2.0
                feature[t, node_id] = f
                split[t, node_id] = s
                left_id, right_id = 2 * node_id + 1, 2 * node_id + 2
                if left_id < self.n_internal:
                    left_b = b.copy()
                    left_b[f, 1] = s
                    pending[left_id] = left_b
                if right_id < self.n_internal:
                    right_b = b.copy()
                    right_b[f, 0] = s
                    pending[right_id] = right_b
        self.feature = feature
        self.split = split

    def _leaf_indices(self, X: np.ndarray) -> np.ndarray:
        """Vectorized routing: (n_trees, n_rows) leaf index per (tree, row) pair, computed
        with `max_depth` numpy operations over the whole batch at once instead of a
        per-row Python tree walk.
        """
        n_rows = X.shape[0]
        tree_idx = np.arange(self.n_trees)[:, None]
        row_idx = np.arange(n_rows)[None, :]
        node = np.zeros((self.n_trees, n_rows), dtype=np.int64)
        for _ in range(self.max_depth):
            f = self.feature[tree_idx, node]
            s = self.split[tree_idx, node]
            go_right = X[row_idx, f] >= s
            node = node * 2 + 1 + go_right.astype(np.int64)
        return node - self.n_internal

    def fit(self, X: np.ndarray) -> "HalfSpaceTrees":
        """Builds tree structure from `X`'s per-channel range, then treats `X` itself as
        the first streaming batch via `partial_fit` -- matching `AnomalyDetector.fit`'s
        contract (nominal-only data) while still exercising the online mass-profile path.
        """
        X = np.asarray(X, dtype=np.float64)
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        span = np.maximum(maxs - mins, 1e-9)
        margin = span * self.size_limit_multiplier * self._rng.uniform(size=X.shape[1])
        bounds = np.stack([mins - margin, maxs + margin], axis=1)
        self._build_trees(bounds)
        self.ref_mass = np.zeros((self.n_trees, self.n_leaves), dtype=np.int64)
        self.latest_mass = np.zeros((self.n_trees, self.n_leaves), dtype=np.int64)
        self._seen_in_window = 0
        self.partial_fit(X)
        return self

    def partial_fit(self, X: np.ndarray) -> "HalfSpaceTrees":
        """Stream new nominal rows into the mass profile without discarding tree
        structure or previously accumulated `ref_mass` -- the incremental alternative to
        a full `fit()` refit when new nominal data (e.g. the next walk-forward fold)
        arrives. Processed in `window_size`-sized chunks so a window swap lands exactly
        on the same row boundary a true row-by-row stream would produce, however large
        the caller's batch is.
        """
        if self.feature is None:
            raise RuntimeError("HalfSpaceTrees must be fit() before partial_fit()")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        pos = 0
        while pos < n:
            take = min(self.window_size - self._seen_in_window, n - pos)
            leaf = self._leaf_indices(X[pos : pos + take])
            for t in range(self.n_trees):
                self.latest_mass[t] += np.bincount(leaf[t], minlength=self.n_leaves)
            self._seen_in_window += take
            pos += take
            if self._seen_in_window >= self.window_size:
                self.ref_mass = self.latest_mass.copy()
                self.latest_mass = np.zeros_like(self.latest_mass)
                self._seen_in_window = 0
        return self

    def flush_window(self) -> "HalfSpaceTrees":
        """Force whatever's accumulated in `latest_mass` to become `ref_mass`
        immediately, regardless of `window_size`.

        `window_size` is a natural knob for genuine row-by-row streaming (a few hundred
        readings), but a caller that wants to treat one arbitrarily-sized batch -- a
        whole month's fold, or a whole training split -- as exactly one reference window
        would otherwise have to pick a `window_size` that happens to divide that batch
        size evenly: a `window_size` *smaller* than the batch silently discards all but
        its last window-size rows, since only the final swap's `latest_mass` survives
        into `ref_mass`. The pattern to use instead: set `window_size` larger than any
        single batch so `partial_fit` never auto-swaps, then call `flush_window()`
        explicitly once per logical batch.
        """
        if self.feature is None:
            raise RuntimeError("HalfSpaceTrees must be fit() before flush_window()")
        self.ref_mass = self.latest_mass.copy()
        self.latest_mass = np.zeros_like(self.latest_mass)
        self._seen_in_window = 0
        return self

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "HalfSpaceTrees":
        """Set a usable threshold from a percentile of held-out nominal scores, mirroring
        `PCAReconstructor.calibrate_threshold` -- kept separate from `partial_fit` so
        calibration data doesn't also feed the mass profile.
        """
        self.threshold_ = float(np.percentile(self.score(validation_values), percentile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        leaf = self._leaf_indices(X)  # (n_trees, n_rows)
        total = np.zeros(X.shape[0])
        for t in range(self.n_trees):
            total += self.ref_mass[t][leaf[t]]
        return -total  # low mass at a leaf => isolated quickly => anomalous

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise RuntimeError("No threshold: pass one explicitly or call calibrate_threshold() first")
        return self.score(X) > thr
