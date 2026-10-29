from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.heavy.autoencoder import _AutoencoderNet


class PULagAutoencoder(AnomalyDetector):
    """Autoencoder extended with two optional exemplar sets, both built from real
    labels revealed after a realistic review lag, never from the model's own
    pseudo-labels.

    - positive: rows from confirmed real anomalies. Pushes their reconstruction
      error above the current batch's error by a margin.
    - hard_negative: rows the model alarmed on that a later review confirmed
      were false positives. Used only by the active-learning variant. Folded
      into the ordinary reconstruction loss with extra weight, so the model
      learns to reconstruct the patterns that fooled it.

    Both arguments are optional and default to plain Autoencoder behavior when
    omitted, which covers every early fold before any review has happened yet.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_sizes: Sequence[int] = (32, 16),
        latent_size: int = 8,
        margin: float = 1.0,
        pu_weight: float = 1.0,
        hard_negative_repeat: int = 2,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.device = device
        self.seed = seed
        self.margin = margin
        self.pu_weight = pu_weight
        self.hard_negative_repeat = hard_negative_repeat
        torch.manual_seed(seed)
        self.net = _AutoencoderNet(n_channels, hidden_sizes, latent_size).to(device)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(
        self,
        X: np.ndarray,
        positive: Optional[np.ndarray] = None,
        hard_negative: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> "PULagAutoencoder":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train(X, positive, hard_negative, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        positive: Optional[np.ndarray] = None,
        hard_negative: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> "PULagAutoencoder":
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            self.std_ = std
        self._train(X, positive, hard_negative, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def _train(
        self,
        X: np.ndarray,
        positive: Optional[np.ndarray],
        hard_negative: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        lr: float,
        verbose: bool,
    ) -> None:
        if hard_negative is not None and len(hard_negative) > 0:
            hn = np.asarray(hard_negative, dtype=np.float64)
            X = np.concatenate([X] + [hn] * self.hard_negative_repeat, axis=0)

        X_t = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        pos_t = None
        if positive is not None and len(positive) > 0:
            pos_t = torch.tensor(self._standardize(np.asarray(positive, dtype=np.float64)), dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = X_t.shape[0]
        self.net.train()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for epoch in range(epochs):
            perm = torch.randperm(n, generator=generator)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                recon = self.net(X_t[idx])
                unlabeled_err = ((recon - X_t[idx]) ** 2).mean(dim=1)
                loss = unlabeled_err.mean()
                if pos_t is not None:
                    pos_recon = self.net(pos_t)
                    pos_err = ((pos_recon - pos_t) ** 2).mean(dim=1)
                    hinge = torch.relu(self.margin + unlabeled_err.mean().detach() - pos_err.mean())
                    loss = loss + self.pu_weight * hinge
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.shape[0]
            if verbose:
                print(f"[PULagAutoencoder] epoch {epoch + 1}/{epochs} loss={total_loss / n:.5f}")

    def score(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("PULagAutoencoder must be fit before scoring")
        self.net.eval()
        X_std = self._standardize(X)
        n = X_std.shape[0]
        errors = np.zeros(n)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_t = torch.tensor(X_std[start:end], dtype=torch.float32, device=self.device)
                recon = self.net(X_t).cpu().numpy()
                errors[start:end] = np.sqrt(((X_std[start:end] - recon) ** 2).mean(axis=1))
        return errors

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "PULagAutoencoder":
        self.threshold_ = float(np.percentile(self.score(validation_values), percentile))
        return self
