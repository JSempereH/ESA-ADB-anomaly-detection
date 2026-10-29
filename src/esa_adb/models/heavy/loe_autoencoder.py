"""Latent Outlier Exposure for a tabular, multivariate telemetry autoencoder."""

from __future__ import annotations

import numpy as np
import torch

from esa_adb.models.heavy.autoencoder import Autoencoder


class LOEAutoencoder(Autoencoder):
    """Autoencoder trained with the hard or soft LOE objective from Qiu et al. (2022).

    Labels are never accepted by this class. Each mini-batch assigns its largest
    ``contamination`` fraction of ``L_n - L_a`` scores to the latent anomaly group.
    The normal loss is per-row reconstruction MSE and the bounded opposing loss is
    ``L_a = max(0, margin - L_n)``. At inference, as required by LOE, only ``L_n``
    (the reconstruction score inherited from :class:`Autoencoder`) is used.
    """

    def __init__(
        self,
        n_channels: int,
        assignment: str,
        contamination: float = 0.10,
        margin: float = 0.50,
        warmup_epochs: int = 3,
        **kwargs,
    ) -> None:
        if assignment not in {"hard", "soft"}:
            raise ValueError("assignment must be 'hard' or 'soft'")
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        super().__init__(n_channels=n_channels, **kwargs)
        self.assignment = assignment
        self.contamination = contamination
        self.margin = margin
        self.warmup_epochs = warmup_epochs

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "LOEAutoencoder":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self._train_loe(X, epochs=epochs, batch_size=batch_size, lr=lr, warmup=True, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3, verbose: bool = True) -> "LOEAutoencoder":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        self._train_loe(X, epochs=epochs, batch_size=batch_size, lr=lr, warmup=False, verbose=verbose)
        return self

    def _train_loe(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, warmup: bool, verbose: bool) -> None:
        values = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.net.train()
        for epoch in range(epochs):
            order = torch.randperm(values.shape[0], generator=generator, device=self.device)
            total_loss = 0.0
            selected_fractions = []
            for start in range(0, values.shape[0], batch_size):
                batch = values[order[start : start + batch_size]]
                normal_loss = (self.net(batch) - batch).square().mean(dim=1)
                if warmup and epoch < self.warmup_epochs:
                    loss = normal_loss.mean()
                    selected_fractions.append(0.0)
                else:
                    anomaly_loss = torch.relu(self.margin - normal_loss)
                    n_anomaly = max(1, round(self.contamination * normal_loss.numel()))
                    n_anomaly = min(n_anomaly, normal_loss.numel() - 1)
                    selected = torch.topk(normal_loss - anomaly_loss, n_anomaly, largest=True).indices
                    if self.assignment == "hard":
                        loss_per_row = normal_loss.clone()
                        loss_per_row[selected] = anomaly_loss[selected]
                    else:
                        loss_per_row = normal_loss.clone()
                        loss_per_row[selected] = 0.5 * (normal_loss[selected] + anomaly_loss[selected])
                    loss = loss_per_row.mean()
                    selected_fractions.append(n_anomaly / normal_loss.numel())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * batch.shape[0]
            if verbose:
                print(
                    f"[LOEAutoencoder-{self.assignment}] epoch {epoch + 1}/{epochs} "
                    f"loss={total_loss / values.shape[0]:.5f} latent-anomaly={np.mean(selected_fractions):.3f}"
                )
