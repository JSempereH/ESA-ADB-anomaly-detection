"""RoCA, following the paper's full architecture (Mou et al., 2025, arXiv:2503.18385).

The follow-up to roca_lite.py's tabular MLP approximation. This version uses the
paper's actual backbone: a TCN encoder over sliding windows, then a Seq2Seq LSTM
for the sequence-contrast pair (a window's own representation vs. its
reconstruction). Scored against a fixed one-class center the same way roca_lite
is, with the same loss terms. Only how a row becomes a feature vector changed.

Two things the paper's text leaves open:
- The TCN+LSTM pipeline outputs a sequence of representations, one per pooled
  timestep, but the center and projections need a single vector. We mean-pool
  the sequence before projecting.
- Window length is a multiple of 8, to match three stride-2 max-pools, picked
  for this dataset's resolution rather than taken from the paper.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector


class _TCNEncoder(nn.Module):
    def __init__(self, n_channels: int, hidden: int, n_blocks: int = 3):
        super().__init__()
        blocks = []
        in_ch = n_channels
        for i in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1) if i == 0 else nn.Identity(),
                    nn.MaxPool1d(2),
                )
            )
            in_ch = hidden
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, w, d)` -> `(B, hidden, L')`."""
        return self.blocks(x.transpose(1, 2))


class _Seq2SeqLSTM(nn.Module):
    def __init__(self, hidden: int, lstm_hidden: int, n_layers: int = 3):
        super().__init__()
        self.encoder = nn.LSTM(hidden, lstm_hidden, num_layers=n_layers, batch_first=True)
        self.decoder = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(lstm_hidden, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, hidden, L')` -> reconstruction `(B, hidden, L')`."""
        b, _, length = z.shape
        seq = z.transpose(1, 2)  # (B, L', hidden)
        _, (h_n, _) = self.encoder(seq)
        context = h_n[-1:].repeat(h_n.shape[0], 1, 1)  # (n_layers, B, lstm_hidden), last-layer context tiled as init state
        dec_in = torch.zeros(b, length, h_n.shape[-1], device=z.device)
        dec_out, _ = self.decoder(dec_in, (context, torch.zeros_like(context)))
        return self.out(dec_out).transpose(1, 2)  # (B, hidden, L')


class RoCAFull(AnomalyDetector):
    def __init__(
        self,
        n_channels: int,
        window_length: int = 24,
        tcn_hidden: int = 16,
        lstm_hidden: int = 16,
        proj_size: int = 16,
        nu: float = 0.07,
        mu: float = 1.0,
        lam: float = 1.0,
        zeta: float = 1.0,
        var_eps: float = 1e-4,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if not 0.0 < nu < 0.5:
            raise ValueError("nu must be in (0, 0.5)")
        if window_length % 8 != 0:
            raise ValueError("window_length must be a multiple of 8 (three stride-2 max-pools)")
        self.n_channels = n_channels
        self.window_length = window_length
        self.nu = nu
        self.mu = mu
        self.lam = lam
        self.zeta = zeta
        self.var_eps = var_eps
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.encoder = _TCNEncoder(n_channels, tcn_hidden).to(device)
        self.seq2seq = _Seq2SeqLSTM(tcn_hidden, lstm_hidden).to(device)
        self.proj = nn.Sequential(nn.Linear(tcn_hidden, proj_size), nn.BatchNorm1d(proj_size), nn.ReLU(), nn.Linear(proj_size, proj_size)).to(device)
        self.center: torch.Tensor | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._generator = torch.Generator(device=device).manual_seed(seed)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _windows(self, values: np.ndarray, stride: int) -> np.ndarray:
        w, d = self.window_length, self.n_channels
        n = values.shape[0]
        if n < w:
            return np.empty((0, w, d))
        windows = np.lib.stride_tricks.sliding_window_view(values, (w, d))[:, 0, :, :]
        return np.ascontiguousarray(windows[::stride])

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)  # (B, hidden, L')
        z_rec = self.seq2seq(z)  # (B, hidden, L')
        q = nn.functional.normalize(self.proj(z.mean(dim=2)), dim=1)
        q2 = nn.functional.normalize(self.proj(z_rec.mean(dim=2)), dim=1)
        return q, q2

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, stride: int = 5, verbose: bool = True) -> "RoCAFull":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.std_ = std
        self.center = None
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def partial_fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3, stride: int = 5, verbose: bool = True) -> "RoCAFull":
        if self.mean_ is None:
            return self.fit(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        self._train(X, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose)
        return self

    def _set_center(self, windows: torch.Tensor, batch_size: int) -> None:
        qs = []
        self.encoder.eval()
        self.seq2seq.eval()
        self.proj.eval()
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch_size):
                q, _ = self._project(windows[start : start + batch_size])
                qs.append(q)
        c = nn.functional.normalize(torch.cat(qs).mean(dim=0), dim=0)
        self.center = c

    def _train(self, X: np.ndarray, epochs: int, batch_size: int, lr: float, stride: int, verbose: bool) -> None:
        windows_np = self._windows(self._standardize(X), stride=stride)
        if windows_np.shape[0] < 2:
            return
        windows = torch.tensor(windows_np, dtype=torch.float32, device=self.device)
        n = windows.shape[0]
        params = list(self.encoder.parameters()) + list(self.seq2seq.parameters()) + list(self.proj.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        if self.center is None:
            self._set_center(windows, batch_size)

        for epoch in range(epochs):
            self.encoder.eval()
            self.seq2seq.eval()
            self.proj.eval()
            with torch.no_grad():
                q_all, q2_all = [], []
                for start in range(0, n, batch_size):
                    q, q2 = self._project(windows[start : start + batch_size])
                    q_all.append(q)
                    q2_all.append(q2)
                q_all, q2_all = torch.cat(q_all), torch.cat(q2_all)
                sim_q, sim_q2 = q_all @ self.center, q2_all @ self.center
                l_inv_all = 2.0 - sim_q - sim_q2
                l_oe_all = 4.0 - l_inv_all
                s_train = l_inv_all - l_oe_all
                n_anom = max(1, math.ceil(self.nu * n))
                y_all = torch.zeros(n, device=self.device)
                y_all[torch.topk(s_train, n_anom).indices] = 1.0

            self.encoder.train()
            self.seq2seq.train()
            self.proj.train()
            order = torch.randperm(n, generator=self._generator, device=self.device)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = windows[idx]
                y = y_all[idx]
                if batch.shape[0] < 2:
                    continue
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
                print(f"[RoCAFull] epoch {epoch + 1}/{epochs} loss={total / n:.5f} labeled_anomalies={int(y_all.sum().item())}")

    def score(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if self.mean_ is None or self.center is None:
            raise RuntimeError("RoCAFull must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        n_rows = X.shape[0]
        w = self.window_length
        scores = np.zeros(n_rows)
        if n_rows < w:
            return scores
        windows_np = self._windows(self._standardize(X), stride=1)
        self.encoder.eval()
        self.seq2seq.eval()
        self.proj.eval()
        per_window = np.empty(windows_np.shape[0])
        with torch.no_grad():
            for start in range(0, windows_np.shape[0], batch_size):
                end = min(start + batch_size, windows_np.shape[0])
                batch = torch.tensor(windows_np[start:end], dtype=torch.float32, device=self.device)
                q, q2 = self._project(batch)
                per_window[start:end] = (2.0 - q @ self.center - q2 @ self.center).cpu().numpy()
        n_windows = per_window.shape[0]
        for k in range(w):
            rows = np.arange(k, k + n_windows)
            np.maximum.at(scores, rows, per_window)
        return scores
