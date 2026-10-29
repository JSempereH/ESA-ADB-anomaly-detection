"""Online-iForest (Leveni et al., ICML 2024, arXiv:2505.09593): isolation trees that learn
and forget incrementally, without periodic retraining.

Follows the paper's structure as described in its full text: each node keeps a bin
height (points routed through it) and a support hyperrectangle; a leaf splits (random
dimension, random value inside its support) once its height reaches eta * 2^depth,
up to depth log2(omega/eta); a sliding window of the last `omega` learned points is
forgotten point by point, and an internal node whose height falls below its split
threshold collapses back into a leaf. Score is 2^(-E[depth] / c), c = log2(omega/eta).

Two approximations, documented here rather than hidden:
- The paper seeds a new child's height by sampling eta * 2^depth points from the bin's
  region; here it's the parent's height times the child's share of the split interval,
  the same uniform-density expectation without storing points.
- `stride` subsamples the stream used for learning (scoring always uses every row),
  because a per-point Python update over a month of 18-second telemetry across 32 trees
  is too slow for the walk-forward benchmark.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from esa_adb.models.base import AnomalyDetector


class _Node:
    __slots__ = ("lo", "hi", "height", "depth", "dim", "val", "left", "right")

    def __init__(self, lo: np.ndarray, hi: np.ndarray, height: float, depth: int) -> None:
        self.lo = lo
        self.hi = hi
        self.height = height
        self.depth = depth
        self.dim = -1
        self.val = 0.0
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class OnlineIsolationForest(AnomalyDetector):
    def __init__(self, n_trees: int = 32, omega: int = 2048, eta: int = 32, stride: int = 10, seed: int = 42) -> None:
        if eta < 1 or omega < eta:
            raise ValueError("require 1 <= eta <= omega")
        self.n_trees = n_trees
        self.omega = omega
        self.eta = eta
        self.stride = stride
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._trees: Optional[list[_Node]] = None
        self._window: deque[np.ndarray] = deque()
        self._c = float(np.log2(omega / eta))
        self._max_depth = max(1, int(np.floor(self._c)))

    def _threshold(self, depth: int) -> float:
        return self.eta * (2.0**depth)

    def _init_trees(self, X: np.ndarray) -> None:
        lo = X.min(axis=0).astype(np.float64)
        hi = X.max(axis=0).astype(np.float64)
        hi = np.where(hi > lo, hi, lo + 1.0)
        self._trees = [_Node(lo.copy(), hi.copy(), 0.0, 0) for _ in range(self.n_trees)]

    def _split(self, node: _Node) -> None:
        dim = int(self._rng.integers(node.lo.shape[0]))
        lo_v, hi_v = float(node.lo[dim]), float(node.hi[dim])
        if hi_v > lo_v:
            val = float(self._rng.uniform(lo_v, hi_v))
            frac = (val - lo_v) / (hi_v - lo_v)
        else:
            val, frac = lo_v, 0.5
        prior = max(node.height - 1.0, 0.0)  # exclude the point currently being learned; it descends after the split
        left_hi = node.hi.copy()
        left_hi[dim] = val
        right_lo = node.lo.copy()
        right_lo[dim] = val
        node.left = _Node(node.lo.copy(), left_hi, prior * frac, node.depth + 1)
        node.right = _Node(right_lo, node.hi.copy(), prior * (1.0 - frac), node.depth + 1)
        node.dim, node.val = dim, val

    def _learn_one(self, root: _Node, x: np.ndarray) -> None:
        node = root
        while True:
            node.height += 1.0
            np.minimum(node.lo, x, out=node.lo)
            np.maximum(node.hi, x, out=node.hi)
            if node.is_leaf:
                if node.height >= self._threshold(node.depth) and node.depth < self._max_depth:
                    self._split(node)
                else:
                    return
            node = node.left if x[node.dim] < node.val else node.right

    def _forget_one(self, root: _Node, x: np.ndarray) -> None:
        node = root
        while True:
            node.height = max(node.height - 1.0, 0.0)
            if node.is_leaf:
                return
            if node.height < self._threshold(node.depth):
                node.left = None
                node.right = None
                return
            node = node.left if x[node.dim] < node.val else node.right

    def _learn(self, X: np.ndarray) -> None:
        for x in X[:: self.stride]:
            x = np.asarray(x, dtype=np.float64)
            for tree in self._trees:
                self._learn_one(tree, x)
            self._window.append(x)
            if len(self._window) > self.omega:
                old = self._window.popleft()
                for tree in self._trees:
                    self._forget_one(tree, old)

    def fit(self, X: np.ndarray) -> "OnlineIsolationForest":
        X = np.asarray(X, dtype=np.float64)
        self._trees = None
        self._window.clear()
        self._rng = np.random.default_rng(self.seed)
        self._init_trees(X)
        self._learn(X)
        return self

    def partial_fit(self, X: np.ndarray) -> "OnlineIsolationForest":
        X = np.asarray(X, dtype=np.float64)
        if self._trees is None:
            return self.fit(X)
        self._learn(X)
        return self

    @staticmethod
    def _flatten(root: _Node) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nodes: list[_Node] = [root]
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if not n.is_leaf:
                nodes.append(n.left)
                nodes.append(n.right)
            i += 1
        index = {id(n): k for k, n in enumerate(nodes)}
        dim = np.full(len(nodes), -1, dtype=np.int64)
        val = np.zeros(len(nodes))
        left = np.zeros(len(nodes), dtype=np.int64)
        right = np.zeros(len(nodes), dtype=np.int64)
        depth = np.zeros(len(nodes), dtype=np.int64)
        for k, n in enumerate(nodes):
            depth[k] = n.depth
            if not n.is_leaf:
                dim[k], val[k] = n.dim, n.val
                left[k], right[k] = index[id(n.left)], index[id(n.right)]
        return dim, val, left, right, depth

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._trees is None:
            raise RuntimeError("OnlineIsolationForest must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        total = np.zeros(X.shape[0])
        for tree in self._trees:
            dim, val, left, right, depth = self._flatten(tree)
            idx = np.zeros(X.shape[0], dtype=np.int64)
            while True:
                rows = np.nonzero(dim[idx] >= 0)[0]
                if rows.size == 0:
                    break
                cur = idx[rows]
                go_left = X[rows, dim[cur]] < val[cur]
                idx[rows] = np.where(go_left, left[cur], right[cur])
            total += depth[idx]
        mean_depth = total / len(self._trees)
        return 2.0 ** (-mean_depth / self._c)
