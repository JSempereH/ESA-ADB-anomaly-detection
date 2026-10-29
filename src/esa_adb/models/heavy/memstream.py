"""Memory-augmented streaming detector, after Bhatia et al. (WWW 2022, MemStream,
arXiv:2106.03837). Follows the paper's architecture (an encoder/decoder pretrained
once, then a frozen encoder with a bounded memory bank that absorbs drift with no
further gradient descent), not a line-by-line port of its exact equations. Treat
it as a reasonable approximation of the paper's mechanism, not a reproduction.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.heavy.autoencoder import _AutoencoderNet


class MemStream(AnomalyDetector):
    """Bounded-memory nearest-neighbor detector over a frozen encoder's latent space.

    The encoder/decoder is trained once, during fit(), and never touched again.
    partial_fit() does no backprop. Instead, a bounded memory bank of latent
    vectors, seeded from the fit batch, absorbs drift by nearest-neighbor
    replacement. A streaming point only gets written to memory if it already
    scores as normal enough. That gating is what gives this model contamination
    tolerance, without a contamination hyperparameter or a pre-fit rejection step.

    score() is the distance, in latent space, from a row's encoding to its
    nearest memory entry. No gradients, which is why partial_fit is cheap
    enough to run every fold.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_sizes: Sequence[int] = (32, 16),
        latent_size: int = 8,
        mem_size: int = 2048,
        beta_percentile: float = 95.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if mem_size < 1:
            raise ValueError("mem_size must be positive")
        self.device = device
        self.seed = seed
        self.latent_size = latent_size
        self.mem_size = mem_size
        self.beta_percentile = beta_percentile
        torch.manual_seed(seed)
        self.net = _AutoencoderNet(n_channels, hidden_sizes, latent_size).to(device)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.memory_: Optional[np.ndarray] = None  # (mem_size, latent_size)
        self.beta_: Optional[float] = None
        self._rng = np.random.default_rng(seed)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _encode(self, X_std: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        self.net.eval()
        out = np.empty((X_std.shape[0], self.latent_size))
        with torch.no_grad():
            for start in range(0, X_std.shape[0], batch_size):
                end = min(start + batch_size, X_std.shape[0])
                X_t = torch.tensor(X_std[start:end], dtype=torch.float32, device=self.device)
                out[start:end] = self.net.encoder(X_t).cpu().numpy()
        return out

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, verbose: bool = True) -> "MemStream":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._pretrain(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)

        latents = self._encode(self._standardize(X))
        n_seed = min(self.mem_size, latents.shape[0])
        seed_idx = self._rng.choice(latents.shape[0], size=n_seed, replace=False)
        memory = latents[seed_idx]
        if n_seed < self.mem_size:
            # Fit batch smaller than mem_size (early folds): pad by resampling with
            # replacement rather than leaving memory undersized, at the cost of some
            # duplicate entries until enough real rows arrive to fill it out.
            pad_idx = self._rng.choice(latents.shape[0], size=self.mem_size - n_seed, replace=True)
            memory = np.concatenate([memory, latents[pad_idx]])
        self.memory_ = memory

        # beta_ is how far a point may sit from its nearest memory entry and still
        # count as normal enough to remember, calibrated from the fit batch's own
        # distances rather than a fixed constant.
        seed_scores = self._nearest_distance(latents)
        self.beta_ = float(np.percentile(seed_scores, self.beta_percentile))
        if verbose:
            print(f"[MemStream] pretrain done, mem_size={self.mem_size}, beta={self.beta_:.5f}")
        return self

    def _pretrain(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, verbose: bool) -> None:
        X_t = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
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
                loss = loss_fn(recon, X_t[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.shape[0]
            if verbose:
                print(f"[MemStream] pretrain epoch {epoch + 1}/{epochs} loss={total_loss / n:.5f}")

    def _nearest_distance(self, latents: np.ndarray, batch_size: int = 5_000) -> np.ndarray:
        """Per-row Euclidean distance to the nearest memory entry, batched so the
        `(batch, mem_size)` distance matrix never spans a whole eval set at once."""
        assert self.memory_ is not None, "memory_ must be seeded before scoring (call fit() first)"
        out = np.empty(latents.shape[0])
        mem = self.memory_
        mem_sq = (mem**2).sum(axis=1)
        for start in range(0, latents.shape[0], batch_size):
            end = min(start + batch_size, latents.shape[0])
            chunk = latents[start:end]
            d2 = (chunk**2).sum(axis=1, keepdims=True) + mem_sq[None, :] - 2 * chunk @ mem.T
            out[start:end] = np.sqrt(np.maximum(d2, 0.0).min(axis=1))
        return out

    def partial_fit(self, X: np.ndarray, batch_size: int = 2_000, verbose: bool = True) -> "MemStream":
        """No gradient step, only the memory bank moves. Processed in chunks so the
        distance matrix stays small and each chunk's admitted rows are written to
        memory before the next chunk is scored, which approximates true row-by-row
        streaming without looping over every row in Python."""
        if self.memory_ is None:
            return self.fit(X, batch_size=min(batch_size, 256), verbose=verbose)
        X = np.asarray(X, dtype=np.float64)
        latents = self._encode(self._standardize(X))
        mem = self.memory_
        n_admitted = 0
        for start in range(0, latents.shape[0], batch_size):
            chunk = latents[start : start + batch_size]
            mem_sq = (mem**2).sum(axis=1)
            d2 = (chunk**2).sum(axis=1, keepdims=True) + mem_sq[None, :] - 2 * chunk @ mem.T
            nearest = np.argmin(d2, axis=1)
            dist = np.sqrt(np.maximum(d2[np.arange(chunk.shape[0]), nearest], 0.0))
            admit = dist < self.beta_
            # If two rows in a chunk map to the same memory slot, the last one
            # wins. That approximates sequential, one-at-a-time replacement.
            mem[nearest[admit]] = chunk[admit]
            n_admitted += int(admit.sum())
        if verbose:
            print(f"[MemStream] partial_fit: {n_admitted}/{latents.shape[0]} rows admitted to memory")
        return self

    def score(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        if self.memory_ is None:
            raise RuntimeError("MemStream must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        latents = self._encode(self._standardize(X), batch_size=batch_size)
        return self._nearest_distance(latents, batch_size=min(batch_size, 5_000))
