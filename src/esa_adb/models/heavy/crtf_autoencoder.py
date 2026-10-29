"""Contamination-resilient autoencoder, after CLEANet's CRTF component (Zhang et
al., 2025, arXiv:2510.22619). Only the paper's abstract was available, not the
full text, so the two named mechanisms (adaptive reconstruction weighting and
clustering-guided contrastive learning) are a good-faith approximation of what
those names describe, not equations taken from the source.

Unlike LOEAutoencoder, which assigns the worst fraction of rows to an "anomaly
loss" each minibatch with a fixed contamination rate, this version computes a
continuous per-row weight from the row's own reconstruction residual, with no
rate to tune. It also adds a clustering term: rows are pulled toward their
nearest k-means centroid in latent space, weighted the same way, so confident
rows reinforce a compact cluster and high-residual rows are excused from it.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from esa_adb.models.heavy.autoencoder import Autoencoder


class CRTFAutoencoder(Autoencoder):
    def __init__(
        self,
        n_channels: int,
        n_clusters: int = 8,
        contrastive_weight: float = 0.5,
        weight_temperature: float = 1.0,
        warmup_epochs: int = 3,
        **kwargs,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if weight_temperature <= 0:
            raise ValueError("weight_temperature must be positive")
        super().__init__(n_channels=n_channels, **kwargs)
        self.n_clusters = n_clusters
        self.contrastive_weight = contrastive_weight
        self.weight_temperature = weight_temperature
        self.warmup_epochs = warmup_epochs

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "CRTFAutoencoder":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train_crtf(X, epochs=epochs, batch_size=batch_size, lr=lr, warmup=True, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "CRTFAutoencoder":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        self._train_crtf(X, epochs=epochs, batch_size=batch_size, lr=lr, warmup=False, verbose=verbose)
        return self

    def _adaptive_weights(self, residual: np.ndarray) -> np.ndarray:
        """Continuous down-weighting of high-residual rows: a sigmoid centered on the
        batch's own median residual (robust to scale drift across folds), scaled by
        `weight_temperature` (larger = softer transition, closer to uniform weights).
        No fixed contamination-rate hyperparameter, unlike LOE."""
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(1.4826 * mad, 1e-8) * self.weight_temperature
        z = (residual - np.median(residual)) / scale
        return 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))

    def _train_crtf(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, warmup: bool, verbose: bool) -> None:
        values = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        n = values.shape[0]
        n_clusters = min(self.n_clusters, n)

        for epoch in range(epochs):
            # Recompute weights and clusters from the current net at the start of
            # each epoch, so both adapt as the net's own notion of normal improves.
            self.net.eval()
            with torch.no_grad():
                latents = self.net.encoder(values)
                recon = self.net.decoder(latents)
                residual = (recon - values).square().mean(dim=1).cpu().numpy()
            centroids_t = None
            assignments = None
            if not (warmup and epoch < self.warmup_epochs):
                weights = self._adaptive_weights(residual)
                km = KMeans(n_clusters=n_clusters, n_init=3, random_state=self.seed)
                assignments = km.fit_predict(latents.cpu().numpy(), sample_weight=weights)
                centroids_t = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
                assignments = torch.tensor(assignments, dtype=torch.long, device=self.device)
                weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
            else:
                weights_t = torch.ones(n, device=self.device)

            self.net.train()
            order = torch.randperm(n, generator=generator, device=self.device)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = values[idx]
                batch_w = weights_t[idx]
                recon_batch = self.net(batch)
                mse = (recon_batch - batch).square().mean(dim=1)
                loss = (batch_w * mse).sum() / batch_w.sum().clamp_min(1e-8)
                if centroids_t is not None:
                    assert assignments is not None
                    latent_batch = self.net.encoder(batch)
                    own_centroid = centroids_t[assignments[idx]]
                    pull = (latent_batch - own_centroid).square().mean(dim=1)
                    loss = loss + self.contrastive_weight * (batch_w * pull).sum() / batch_w.sum().clamp_min(1e-8)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * idx.shape[0]
            if verbose:
                print(f"[CRTFAutoencoder] epoch {epoch + 1}/{epochs} loss={total_loss / n:.5f} mean_weight={weights_t.mean().item():.3f}")
