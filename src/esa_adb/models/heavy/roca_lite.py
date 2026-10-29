"""RoCA-lite: the three-term objective of RoCA (Mou et al., 2025, arXiv:2503.18385) on a
tabular MLP instead of the paper's TCN + Seq2Seq LSTM backbone. Not a faithful port:

- Encoder/decoder are this project's MLP autoencoder (no temporal convolutions) and
  the "sequence contrast" pair is the representation of a row and the representation
  of its own reconstruction (encoder applied to decoder(encoder(x))), rather than an
  LSTM summary of the sequence.
- One-class center Ce is fixed after the first fit, following the paper's rule of only
  updating it during initial epochs.
- Latent anomaly labels: each epoch the top ceil(nu * N) rows by S = L_inv - L_OE are
  labeled anomalous (y = 1). nu is the contamination fraction, fixed by the caller
  (default 0.07 = Mission 2's measured non-nominal share of the training split).
- The variance term is read as a per-dimension hinge on the batch standard deviation of
  the projected features, max(0, zeta - std), since the paper's per-sample notation is
  ambiguous on this point.

Scoring uses only the invariance term, as in the paper's inference rule.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.heavy.autoencoder import _AutoencoderNet


class RoCALite(AnomalyDetector):
    def __init__(
        self,
        n_channels: int,
        nu: float = 0.07,
        hidden_sizes: tuple[int, ...] = (32, 16),
        latent_size: int = 8,
        proj_size: int = 16,
        mu: float = 1.0,
        lam: float = 1.0,
        zeta: float = 1.0,
        var_eps: float = 1e-4,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if not 0.0 < nu < 0.5:
            raise ValueError("nu must be in (0, 0.5)")
        self.nu = nu
        self.mu = mu
        self.lam = lam
        self.zeta = zeta
        self.var_eps = var_eps
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.net = _AutoencoderNet(n_channels, hidden_sizes, latent_size).to(device)
        self.proj = nn.Sequential(nn.Linear(latent_size, proj_size), nn.ReLU(), nn.Linear(proj_size, proj_size)).to(device)
        self.center: torch.Tensor | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._generator = torch.Generator(device=device).manual_seed(seed)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.net.encoder(x)
        z_rec = self.net.encoder(self.net.decoder(z))
        q = nn.functional.normalize(self.proj(z), dim=1)
        q2 = nn.functional.normalize(self.proj(z_rec), dim=1)
        return q, q2

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "RoCALite":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self.center = None
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "RoCALite":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def _set_center(self, values: torch.Tensor) -> None:
        with torch.no_grad():
            q, _ = self._project(values)
            c = nn.functional.normalize(q.mean(dim=0), dim=0)
        self.center = c

    def _train(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, verbose: bool) -> None:
        values = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        n = values.shape[0]
        n_anom = max(1, math.ceil(self.nu * n))
        params = list(self.net.parameters()) + list(self.proj.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        if self.center is None:
            self._set_center(values)
        for epoch in range(epochs):
            with torch.no_grad():
                self.net.eval()
                q_all, q2_all = self._project(values)
                sim_q = q_all @ self.center
                sim_q2 = q2_all @ self.center
                l_inv_all = 2.0 - sim_q - sim_q2
                l_oe_all = 4.0 - l_inv_all
                s_train = l_inv_all - l_oe_all
                y_all = torch.zeros(n, device=self.device)
                y_all[torch.topk(s_train, n_anom).indices] = 1.0

            self.net.train()
            self.proj.train()
            order = torch.randperm(n, generator=self._generator, device=self.device)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = values[idx]
                y = y_all[idx]
                q, q2 = self._project(batch)
                l_inv = 2.0 - q @ self.center - q2 @ self.center
                l_oe = 4.0 - l_inv
                joint = (self.mu * y * l_oe + (1.0 - y) * l_inv).mean()
                var_term = (
                    torch.relu(self.zeta - torch.sqrt(q.var(dim=0, unbiased=False) + self.var_eps)).mean()
                    + torch.relu(self.zeta - torch.sqrt(q2.var(dim=0, unbiased=False) + self.var_eps)).mean()
                )
                loss = joint + 0.5 * self.lam * var_term
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                total += loss.item() * idx.shape[0]
            if verbose:
                print(f"[RoCALite] epoch {epoch + 1}/{epochs} loss={total / n:.5f} labeled_anomalies={int(y_all.sum().item())}")

    def score(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        if self.mean_ is None or self.center is None:
            raise RuntimeError("RoCALite must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0])
        self.net.eval()
        self.proj.eval()
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                x = torch.tensor(self._standardize(X[start:end]), dtype=torch.float32, device=self.device)
                q, q2 = self._project(x)
                out[start:end] = (2.0 - q @ self.center - q2 @ self.center).cpu().numpy()
        return out
