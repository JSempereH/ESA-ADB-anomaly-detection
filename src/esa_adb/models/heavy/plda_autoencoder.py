"""PLDA-lite: a parameter-behavior sample weighting for the autoencoder, after Zhang et al.
(TKDE 2024, "Angel or Devil", arXiv:2410.21322). NOT the full method: the paper's
parameter-behavior term is |H^{-1} grad_theta L|, and its augmentation is a DQN agent
over sliding-window actions. Here (a) the Hessian is replaced by the identity, so the
behavior is the raw per-sample gradient norm, and (b) the augmentation is replaced by a
fixed 2-D rule on the batch's own loss and behavior ranks.

Parameter behavior uses the closed-form gradient of the per-row MSE w.r.t. the final
decoder layer W: dL/dW = (2/d) r h^T with r = reconstruction residual and h the input to
that layer, so ||dL/dW|| = (2/d) ||r|| ||h||. That makes the signal one matrix product
per epoch, not a backward pass per sample.

Weighting, per row, from its loss rank L and behavior rank P (both in [0, 1]):
    w = 1 + b * L * (1 - P) - a * L * P, clipped to [0.05, 2]
High loss with high behavior (the paper's "anomaly contamination") is down-weighted;
high loss with low behavior (the paper's "hard normal") is up-weighted; low-loss rows
stay at weight 1. Weights are recomputed each epoch from the current net.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from esa_adb.models.heavy.autoencoder import Autoencoder


def _rank_percentile(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.shape[0])
    ranks[order] = np.arange(x.shape[0])
    return ranks / max(x.shape[0] - 1, 1)


def plda_weights(loss: np.ndarray, behavior: np.ndarray, a: float = 0.9, b: float = 0.5) -> np.ndarray:
    L = _rank_percentile(loss)
    P = _rank_percentile(behavior)
    w = 1.0 + b * L * (1.0 - P) - a * L * P
    return np.clip(w, 0.05, 2.0)


class PLDAAutoencoder(Autoencoder):
    def __init__(
        self,
        n_channels: int,
        ac_penalty: float = 0.9,
        hn_boost: float = 0.5,
        warmup_epochs: int = 3,
        hidden_sizes: Sequence[int] = (32, 16),
        **kwargs,
    ) -> None:
        super().__init__(n_channels=n_channels, hidden_sizes=hidden_sizes, **kwargs)
        self.ac_penalty = ac_penalty
        self.hn_boost = hn_boost
        self.warmup_epochs = warmup_epochs

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "PLDAAutoencoder":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train_plda(X, epochs=epochs, batch_size=batch_size, lr=lr, warmup=True, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "PLDAAutoencoder":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        self._train_plda(X, epochs=epochs, batch_size=batch_size, lr=lr, warmup=False, verbose=verbose)
        return self

    def _signals(self, values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        dec = self.net.decoder
        with torch.no_grad():
            z = self.net.encoder(values)
            h = dec[:-1](z)
            recon = dec[-1](h)
            resid = recon - values
            d = values.shape[1]
            loss = resid.square().mean(dim=1).cpu().numpy()
            behavior = ((2.0 / d) * resid.norm(dim=1) * h.norm(dim=1)).cpu().numpy()
        return loss, behavior

    def _train_plda(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, warmup: bool, verbose: bool) -> None:
        values = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        n = values.shape[0]
        for epoch in range(epochs):
            self.net.eval()
            loss_np, behavior_np = self._signals(values)
            if warmup and epoch < self.warmup_epochs:
                weights = np.ones(n)
            else:
                weights = plda_weights(loss_np, behavior_np, a=self.ac_penalty, b=self.hn_boost)
            w_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

            self.net.train()
            order = torch.randperm(n, generator=generator, device=self.device)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = values[idx]
                bw = w_t[idx]
                mse = (self.net(batch) - batch).square().mean(dim=1)
                loss = (bw * mse).sum() / bw.sum().clamp_min(1e-8)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                total += loss.item() * idx.shape[0]
            if verbose:
                print(f"[PLDAAutoencoder] epoch {epoch + 1}/{epochs} loss={total / n:.5f} mean_weight={weights.mean():.3f}")
