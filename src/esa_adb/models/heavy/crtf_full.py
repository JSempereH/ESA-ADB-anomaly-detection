"""CRTF, following CLEANet's full method (Zhang et al., 2025, arXiv:2510.22619),
read from the complete paper text rather than just its abstract. crtf_autoencoder.py
is the earlier, abstract-only approximation; this is the faithful follow-up.

Architecture: a window is encoded two ways, a time encoder shared across
channels (each channel's own shape over time) and a feature encoder shared
across timesteps (cross-channel structure at each instant). Both are stacks of
linear and ReLU layers. Decoding concatenates both encodings and reconstructs
the window with one linear layer.

Training, per minibatch:
- Each window gets a contamination score: mean cosine similarity, in latent
  space, to its nearest neighbors within the minibatch. Higher similarity
  reads as more likely contaminated, since the paper's own contamination
  tends to show up as near-duplicate bursts.
- An adaptive weight from that score down-weights the reconstruction loss for
  windows that look contaminated.
- A nearest-neighbor clustering step, computed on the same minibatch, groups
  windows into clusters. Those clusters give positive pairs (same cluster) and
  negative pairs (the most different other cluster) for a contrastive loss.

Two deviations from the paper: the weight's sigmoid midpoint is the current
minibatch's own median contamination score, since the paper never states how
to pick it. And the decoder is linear, not sigmoid, because this project
z-scores channels instead of scaling them to [0, 1].
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector


def _mlp_stack(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    return nn.Sequential(*layers)


class _ConjugateMLP(nn.Module):
    def __init__(self, n_channels: int, window_length: int, hidden_time: int, hidden_feat: int, n_layers: int):
        super().__init__()
        self.n_channels = n_channels
        self.window_length = window_length
        self.time_encoder = _mlp_stack(window_length, hidden_time, n_layers)  # shared across channels
        self.feature_encoder = _mlp_stack(n_channels, hidden_feat, n_layers)  # shared across timesteps
        self.decoder = nn.Linear(n_channels * hidden_time + window_length * hidden_feat, n_channels * window_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x`: `(B, w, d)`. Returns `(recon (B, w, d), z (B, d*hidden_time + w*hidden_feat))`."""
        b = x.shape[0]
        z_t = self.time_encoder(x.transpose(1, 2))  # (B, d, hidden_time)
        z_f = self.feature_encoder(x)  # (B, w, hidden_feat)
        z = torch.cat([z_t.reshape(b, -1), z_f.reshape(b, -1)], dim=1)
        recon = self.decoder(z).reshape(b, self.window_length, self.n_channels)
        return recon, z


def _finch_clusters(z: np.ndarray) -> np.ndarray:
    """First-neighbor connected-component clustering (CLEANet eq. 6)."""
    n = z.shape[0]
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    d2 = ((zn[:, None, :] - zn[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn1 = d2.argmin(axis=1)

    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        union(i, int(nn1[i]))  # "j = k_i^1"
    groups: dict[int, list[int]] = {}
    for i, j in enumerate(nn1):
        groups.setdefault(int(j), []).append(i)
    for members in groups.values():  # "k_i^1 = k_j^1"
        for a in members[1:]:
            union(members[0], a)
    return np.array([find(i) for i in range(n)])


class CRTFFull(AnomalyDetector):
    def __init__(
        self,
        n_channels: int,
        window_length: int = 20,
        hidden_time: int = 8,
        hidden_feat: int = 8,
        n_layers: int = 2,
        k_neighbors: int = 10,
        alpha: float = 5.0,
        lam: float = 0.5,
        temperature: float = 0.5,
        min_cluster_size: int = 5,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        self.n_channels = n_channels
        self.window_length = window_length
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.lam = lam
        self.temperature = temperature
        self.min_cluster_size = min_cluster_size
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.net = _ConjugateMLP(n_channels, window_length, hidden_time, hidden_feat, n_layers).to(device)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _windows(self, values: np.ndarray, stride: int) -> np.ndarray:
        w, d = self.window_length, self.n_channels
        n = values.shape[0]
        if n < w:
            return np.empty((0, w, d))
        windows = np.lib.stride_tricks.sliding_window_view(values, (w, d))[:, 0, :, :]
        return np.ascontiguousarray(windows[::stride])

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, stride: int = 5, verbose: bool = True) -> "CRTFFull":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3, stride: int = 5, verbose: bool = True) -> "CRTFFull":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def _contrastive_loss(self, z: torch.Tensor) -> torch.Tensor:
        labels = _finch_clusters(z.detach().cpu().numpy())
        uniq, counts = np.unique(labels, return_counts=True)
        valid = uniq[counts >= self.min_cluster_size]
        if valid.shape[0] < 2:
            return torch.zeros((), device=z.device)

        zn = nn.functional.normalize(z, dim=1)
        sim = zn @ zn.T / self.temperature  # (B, B)
        centroids = torch.stack([zn[torch.tensor(labels == c, device=z.device)].mean(dim=0) for c in valid])
        centroid_dist = torch.cdist(centroids, centroids)
        farthest = {int(c): int(valid[int(centroid_dist[i].argmax())]) for i, c in enumerate(valid)}

        losses = []
        for c in valid:
            members = np.nonzero(labels == c)[0]
            neg_members = np.nonzero(labels == farthest[int(c)])[0]
            if members.shape[0] < 2 or neg_members.shape[0] == 0:
                continue
            pos_sim = sim[np.ix_(members, members)]
            pos_mask = ~torch.eye(members.shape[0], dtype=torch.bool, device=z.device)
            pos_exp = torch.exp(pos_sim).masked_select(pos_mask).reshape(members.shape[0], -1).sum(dim=1)
            neg_sim = sim[np.ix_(members, neg_members)]
            neg_exp = torch.exp(neg_sim).sum(dim=1)
            losses.append((-torch.log(pos_exp.clamp_min(1e-12) / neg_exp.clamp_min(1e-12))).mean())
        if not losses:
            return torch.zeros((), device=z.device)
        return torch.stack(losses).mean()

    def _train(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, stride: int, verbose: bool) -> None:
        windows = self._windows(self._standardize(X), stride=stride)
        if windows.shape[0] < 2:
            return
        values = torch.tensor(windows, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        n = values.shape[0]
        self.net.train()
        for epoch in range(epochs):
            order = torch.randperm(n, generator=generator, device=self.device)
            total_awrl, total_contra = 0.0, 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = values[idx]
                if batch.shape[0] < max(4, self.min_cluster_size):
                    continue
                recon, z = self.net(batch)

                with torch.no_grad():
                    zn = nn.functional.normalize(z, dim=1)
                    cos = zn @ zn.T
                    cos.fill_diagonal_(-float("inf"))
                    k = min(self.k_neighbors, batch.shape[0] - 1)
                    cont = cos.topk(k, dim=1).values.mean(dim=1)
                    tau = cont.median()
                    weight = torch.sigmoid(-self.alpha * (cont - tau))

                se = (recon - batch).square().sum(dim=(1, 2))
                l_awrl = (weight * se).mean()
                l_contra = self._contrastive_loss(z)
                loss = l_awrl + self.lam * l_contra

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                total_awrl += l_awrl.item() * batch.shape[0]
                total_contra += float(l_contra.item()) * batch.shape[0]
            if verbose:
                print(f"[CRTFFull] epoch {epoch + 1}/{epochs} AWRL={total_awrl / n:.5f} contra={total_contra / n:.5f}")

    def score(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("CRTFFull must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        n_rows = X.shape[0]
        w = self.window_length
        scores = np.zeros(n_rows)
        if n_rows < w:
            return scores
        windows = self._windows(self._standardize(X), stride=1)  # dense, one per row offset
        self.net.eval()
        per_window = np.empty((windows.shape[0], w))
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch_size):
                end = min(start + batch_size, windows.shape[0])
                batch = torch.tensor(windows[start:end], dtype=torch.float32, device=self.device)
                recon, _ = self.net(batch)
                per_window[start:end] = (recon - batch).square().sum(dim=2).cpu().numpy()
        # Row r is covered, at relative offset k, by the window starting at r-k (for each
        # k in [0, w)): fold those w candidate offsets into one max-over-covering-windows
        # row score with w vectorized passes instead of a per-window Python loop.
        n_windows = per_window.shape[0]
        for k in range(w):
            rows = np.arange(k, k + n_windows)
            np.maximum.at(scores, rows, per_window[:, k])
        return scores
