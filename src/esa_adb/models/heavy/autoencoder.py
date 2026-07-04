from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector


class _AutoencoderNet(nn.Module):
    def __init__(self, n_channels: int, hidden_sizes: Sequence[int], latent_size: int):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        in_size = n_channels
        for h in hidden_sizes:
            encoder_layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        encoder_layers.append(nn.Linear(in_size, latent_size))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_size = latent_size
        for h in reversed(hidden_sizes):
            decoder_layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        decoder_layers.append(nn.Linear(in_size, n_channels))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class Autoencoder(AnomalyDetector):
    """Nonlinear reconstruction-error anomaly detector -- the natural extension of
    `PCAReconstructor` (the exact *linear* special case of this same architecture, with
    no hidden layers and a linear encoder/decoder) once linear reconstruction
    outperformed forecasting-based `LSTMForecaster` on real Mission2 data. Tests whether
    real channel relationships have nonlinear structure a linear PCA subspace can't
    capture.

    Operates row-by-row (not windowed), matching `PCAReconstructor`'s representation, for
    a direct linear-vs-nonlinear comparison rather than confounding the comparison with a
    different input representation too.

    `fit()` trains on nominal rows only (reconstruction MSE). `score()`/`calibrate_threshold()`
    mirror `PCAReconstructor`/`LSTMForecaster`: threshold calibration uses a held-out
    nominal validation set, never training residuals, which systematically underestimate
    real-world error (see `LSTMForecaster`'s docstring).
    """

    def __init__(
        self,
        n_channels: int,
        hidden_sizes: Sequence[int] = (32, 16),
        latent_size: int = 8,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)  # weight init draws from the global RNG at construction time
        self.net = _AutoencoderNet(n_channels, hidden_sizes, latent_size).to(device)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, verbose: bool = True) -> "Autoencoder":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3, verbose: bool = True) -> "Autoencoder":
        """Continue gradient descent on `self.net`'s EXISTING weights using only this new
        batch -- the genuine "update from the new month, don't forget the rest"
        counterpart to `fit()`'s batch retrain. `fit()` never reinitializes `self.net`
        either, so the only real difference here is `mean_`/`std_`: `fit()` recomputes
        them from whatever `X` it's given every time, which would silently redefine
        "normal" every fold; `partial_fit` fixes them once, from the FIRST batch it ever
        sees, and never touches them again.
        """
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            self.std_ = std
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def _train(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, verbose: bool) -> None:
        X_t = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n = X_t.shape[0]
        self.net.train()
        # Reseed here too (not just __init__): makes the epoch-by-epoch shuffling below
        # reproducible on its own, regardless of what other torch code ran between
        # construction and this call and may have advanced the global RNG state.
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
                print(f"[Autoencoder] epoch {epoch + 1}/{epochs} loss={total_loss / n:.5f}")

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "Autoencoder":
        self.threshold_ = float(np.percentile(self.score(validation_values), percentile))
        return self

    def score_per_channel(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        """`(n_rows, n_channels)` squared reconstruction error per channel, in
        standardized units -- same explainability role as
        `PCAReconstructor.score_per_channel`, but over the nonlinear reconstruction.
        """
        if self.mean_ is None:
            raise RuntimeError("Autoencoder must be fit before scoring")
        self.net.eval()
        X_std = self._standardize(X)
        n = X_std.shape[0]
        errors = np.zeros((n, X_std.shape[1]))
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_t = torch.tensor(X_std[start:end], dtype=torch.float32, device=self.device)
                recon = self.net(X_t).cpu().numpy()
                errors[start:end] = (X_std[start:end] - recon) ** 2
        return errors

    def most_anomalous_channels(self, X: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        """For each row, the `top_n` channels with the highest reconstruction error."""
        err = self.score_per_channel(X)
        top_n = min(top_n, err.shape[1])
        order = np.argsort(-err, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def score(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        return np.sqrt(self.score_per_channel(X, batch_size=batch_size).mean(axis=1))

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise RuntimeError("No threshold: pass one explicitly or call calibrate_threshold() first")
        return self.score(X) > thr
