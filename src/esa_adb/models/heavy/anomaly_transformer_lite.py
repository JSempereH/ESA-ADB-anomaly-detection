"""Anomaly Transformer (Xu et al., ICLR 2022, arXiv:2110.02642), read from the
paper's complete text, not just its abstract.

This scores something different from the reconstruction-based models in this
project (Autoencoder, LOE, USDR, MemStream, CRTF, PLDA), which all share a
weak spot: a rare-but-normal event and a real anomaly are both statistically
rare and both hard to reconstruct. This one instead scores how much a point's
attention collapses onto its immediate neighbors instead of forming a longer-
range association with the rest of the window, the Association Discrepancy.
The paper's key finding is that a real anomaly tends to have a lower
discrepancy than normal points, the opposite sign from a residual-based score.

Each layer learns a Gaussian prior association (how much a point should
attend to its neighbors) alongside the usual self-attention. The discrepancy
is the divergence between the two, averaged across layers. Training is a
minimax game: one term pulls the prior toward the series association, another
pushes the series association away from the prior. The score combines the
discrepancy with the usual reconstruction error.

Scaled down from the paper's own defaults to fit this project's CPU training
budget: window 30 instead of 100, d_model 32, 2 layers, 4 heads.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector

_EPS = 1e-6


class _AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.sigma = nn.Linear(d_model, n_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, rel_dist2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x`: `(B, N, d_model)`, `rel_dist2`: precomputed `(N, N)` squared relative
        time-distance `(i-j)^2`. Returns `(out, prior, series)`, `prior`/`series`
        averaged over heads to `(B, N, N)` row-stochastic matrices for the discrepancy."""
        b, n, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q(x).view(b, n, h, hd).transpose(1, 2)  # (B, h, N, hd)
        k = self.k(x).view(b, n, h, hd).transpose(1, 2)
        v = self.v(x).view(b, n, h, hd).transpose(1, 2)
        series = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)  # (B, h, N, N)

        sigma = torch.nn.functional.softplus(self.sigma(x)) + _EPS  # (B, N, h)
        sigma = sigma.transpose(1, 2).unsqueeze(-1)  # (B, h, N, 1)
        gauss = torch.exp(-rel_dist2 / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)
        prior = gauss / gauss.sum(dim=-1, keepdim=True).clamp_min(_EPS)  # (B, h, N, N), rows sum to 1

        z = (series @ v).transpose(1, 2).reshape(b, n, h * hd)
        return self.out(z), prior.mean(dim=1), series.mean(dim=1)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int):
        super().__init__()
        self.attn = _AnomalyAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, rel_dist2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, prior, series = self.attn(x, rel_dist2)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, prior, series


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise discrete KL(p || q), p/q: `(B, N, N)` row-stochastic -> `(B, N)`."""
    return (p * (torch.log(p.clamp_min(_EPS)) - torch.log(q.clamp_min(_EPS)))).sum(dim=-1)


class AnomalyTransformerLite(AnomalyDetector):
    def __init__(
        self,
        n_channels: int,
        window_length: int = 30,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_hidden: int = 64,
        lam: float = 3.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.window_length = window_length
        self.lam = lam
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.embed = nn.Linear(n_channels, d_model).to(device)
        self.layers = nn.ModuleList([_EncoderLayer(d_model, n_heads, ff_hidden) for _ in range(n_layers)]).to(device)
        self.decode = nn.Linear(d_model, n_channels).to(device)
        idx = torch.arange(window_length, dtype=torch.float32)
        self._rel_dist2 = ((idx[:, None] - idx[None, :]) ** 2).to(device)
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

    def _parameters(self):
        return list(self.embed.parameters()) + list(self.layers.parameters()) + list(self.decode.parameters())

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        h = self.embed(x)
        priors, seriess = [], []
        for layer in self.layers:
            h, prior, series = layer(h, self._rel_dist2)
            priors.append(prior)
            seriess.append(series)
        return self.decode(h), priors, seriess

    def _assdis(self, priors: list[torch.Tensor], seriess: list[torch.Tensor], detach_prior: bool, detach_series: bool) -> torch.Tensor:
        total = 0.0
        for p, s in zip(priors, seriess):
            pp = p.detach() if detach_prior else p
            ss = s.detach() if detach_series else s
            total = total + _kl(pp, ss) + _kl(ss, pp)
        return total / len(priors)  # (B, N)

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-4, stride: int = 5, verbose: bool = True) -> "AnomalyTransformerLite":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-4, stride: int = 5, verbose: bool = True) -> "AnomalyTransformerLite":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def _train(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, stride: int, verbose: bool) -> None:
        windows_np = self._windows(self._standardize(X), stride=stride)
        if windows_np.shape[0] < 2:
            return
        windows = torch.tensor(windows_np, dtype=torch.float32, device=self.device)
        n = windows.shape[0]
        optimizer = torch.optim.Adam(self._parameters(), lr=lr)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for epoch in range(epochs):
            order = torch.randperm(n, generator=generator, device=self.device)
            total_recon, total_assdis = 0.0, 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = windows[idx]
                recon, priors, seriess = self._forward(batch)
                recon_loss = (recon - batch).square().mean()
                # Combined minimax loss: the minimize-phase term (prior chases detached
                # series) and the maximize-phase term (series flees detached prior) in
                # one backward pass, each only reaching its own branch's parameters.
                minimize_term = self._assdis(priors, seriess, detach_prior=False, detach_series=True).mean()
                maximize_term = self._assdis(priors, seriess, detach_prior=True, detach_series=False).mean()
                loss = recon_loss + self.lam * minimize_term - self.lam * maximize_term
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._parameters(), max_norm=5.0)
                optimizer.step()
                total_recon += recon_loss.item() * idx.shape[0]
                total_assdis += maximize_term.item() * idx.shape[0]
            if verbose:
                print(f"[AnomalyTransformerLite] epoch {epoch + 1}/{epochs} recon={total_recon / n:.5f} assdis={total_assdis / n:.5f}")

    def score(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("AnomalyTransformerLite must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        n_rows = X.shape[0]
        w = self.window_length
        scores = np.zeros(n_rows)
        if n_rows < w:
            return scores
        windows_np = self._windows(self._standardize(X), stride=1)
        per_window = np.empty((windows_np.shape[0], w))
        with torch.no_grad():
            for start in range(0, windows_np.shape[0], batch_size):
                end = min(start + batch_size, windows_np.shape[0])
                batch = torch.tensor(windows_np[start:end], dtype=torch.float32, device=self.device)
                recon, priors, seriess = self._forward(batch)
                assdis = self._assdis(priors, seriess, detach_prior=False, detach_series=False)  # (b, w)
                weight = torch.softmax(-assdis, dim=-1)
                point_error = (recon - batch).square().sum(dim=-1)  # (b, w)
                per_window[start:end] = (weight * point_error).cpu().numpy()
        n_windows = per_window.shape[0]
        for k in range(w):
            rows = np.arange(k, k + n_windows)
            np.maximum.at(scores, rows, per_window[:, k])
        return scores
