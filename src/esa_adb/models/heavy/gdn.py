from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from esa_adb.models.base import AnomalyDetector
from esa_adb.models.heavy.lstm_forecaster import build_forecast_windows


def _build_graph(embeddings: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k cosine-similarity adjacency from sensor embeddings (Deng & Hooi, AAAI 2021,
    eq. 2-3): sensor i's candidate neighbors are every OTHER sensor, ranked by cosine
    similarity of their embedding vectors, keeping only the top `k`. A self-loop is
    always added afterwards (needed for `_GDNNet.forward`'s softmax, eq. 8: attention is
    normalized over `N(i) union {i}`, not just the neighbors).
    """
    n = embeddings.shape[0]
    norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = norm @ norm.T
    sim.fill_diagonal_(-float("inf"))  # a sensor is never its own "candidate relation"
    k = min(top_k, n - 1)
    topk_idx = sim.topk(k, dim=1).indices
    adjacency = torch.zeros((n, n), dtype=torch.bool, device=embeddings.device)
    adjacency.scatter_(1, topk_idx, True)
    adjacency.fill_diagonal_(True)
    return adjacency


class _GDNNet(nn.Module):
    """Sensor embeddings + graph-attention forecaster (Deng & Hooi, AAAI 2021, eq. 5-9).

    `N` (channel count) is small in this project's missions (<=100), so attention is
    computed as one dense `(N, N)` matrix per batch, masked by the learned top-k
    adjacency -- simpler than a sparse graph representation and no `torch_geometric`
    dependency, at a cost (`O(N^2)`) that's negligible at this scale.
    """

    def __init__(self, n_channels: int, window_length: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.sensor_embeddings = nn.Parameter(torch.randn(n_channels, embed_dim) * 0.1)
        self.W = nn.Linear(window_length, embed_dim, bias=False)  # eq. 6: shared transform of each sensor's own window
        self.attn = nn.Parameter(torch.randn(4 * embed_dim) * 0.1)  # eq. 7: 'a'
        self.output = nn.Sequential(
            nn.Linear(n_channels * embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_channels),
        )
        self.register_buffer("adjacency", torch.eye(n_channels, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x`: `(B, window_length, N)` (matches `build_forecast_windows`'s own
        convention). Returns `(prediction (B, N), attention (B, N, N))` -- the attention
        matrix is the explainability payload: `attention[b, i, j]` is how much sensor
        i's forecast at that instant weighed neighbor j.

        Eq. 7's attention logit is `a^T(g_i (+) g_j)`, i.e. a bilinear form over a
        concatenation -- computed here as `a_i . g_i + a_j . g_j` (splitting `a` into
        its two `2d`-length halves) instead of literally materializing every `(i, j)`
        pair's `4d`-length concatenation first. Both are exactly the same value (dot
        product distributes over concatenation), but the naive version allocates a
        `(B, N, N, 4*embed_dim)` tensor; this version replaces it with two cheap
        `(B, N, embed_dim)` reductions and one `(B, N, N)` broadcast sum, an order of
        magnitude less compute for an identical result.
        """
        x = x.transpose(1, 2)  # (B, N, w)
        b, n, _ = x.shape
        wx = self.W(x)  # (B, N, d) -- eq. 6's Wx_i for every i at once
        v = self.sensor_embeddings.unsqueeze(0).expand(b, -1, -1)  # (B, N, d)
        g = torch.cat([v, wx], dim=-1)  # (B, N, 2d)

        a_i, a_j = self.attn[: 2 * self.embed_dim], self.attn[2 * self.embed_dim :]
        s_i = g @ a_i  # (B, N) -- the g_i-dependent half of a^T(g_i (+) g_j)
        s_j = g @ a_j  # (B, N) -- the g_j-dependent half
        scores = torch.nn.functional.leaky_relu(s_i.unsqueeze(2) + s_j.unsqueeze(1))  # (B, N, N)

        scores = scores.masked_fill(~self.adjacency.unsqueeze(0), float("-inf"))
        attention = torch.softmax(scores, dim=2)  # eq. 8, normalized over N(i) union {i}

        z = torch.relu(torch.einsum("bij,bjd->bid", attention, wx))  # eq. 5 (self+neighbor terms unified into one sum)
        rep = (v * z).reshape(b, n * self.embed_dim)  # eq. 9's [v_1 o z_1, ..., v_N o z_N]
        return self.output(rep), attention


class GDN(AnomalyDetector):
    """Graph Deviation Network (Deng & Hooi, *Graph Neural Network-Based Anomaly
    Detection in Multivariate Time Series*, AAAI 2021) -- learns which channels are
    related to each other (a graph, from sensor embedding similarity) instead of
    assuming they all matter equally (`Autoencoder`) or are linearly related
    (`PCAReconstructor`), then forecasts each channel from an attention-weighted
    combination of its graph neighbors and flags large, per-channel forecast errors.

    Natively explainable by design (the paper's own main selling point over
    `Autoencoder`/`LSTMForecaster`): `graph_edges()` shows which channels the model
    considers related, `score_per_channel()` shows which channel(s) actually drove a
    given anomaly score, and `attention_for_window()` shows which neighbors mattered
    most for one specific forecast.

    Three deliberate deviations from the paper, all documented since none are free:

    0. **Channels are z-scored before anything else.** Not explicit in the paper's own
       text, but standard practice for this literature and already how every other model
       in this project handles channels of wildly different physical scale
       (`PCAReconstructor`/`Autoencoder` both do the same thing, for the same reason,
       "Channels are z-scored..." in `PCAReconstructor`'s own docstring). With 47
       raw-scale channels, MSE training loss would otherwise be dominated by whichever
       few channels happen to have the largest raw variance, starving the rest of useful
       gradient -- their sensor embeddings would never get a real training signal to
       organize into a meaningful similarity structure. See README Explainability
       section for how well the learned graph recovers `channels.csv` subsystem
       structure in practice.
    1. **Normalization stats are calibrated, not self-referential.** The paper recomputes
       each channel's median/IQR error from whatever data is currently being scored (its
       eq. 12) -- convenient for a one-shot benchmark, but self-referential in a way this
       project avoids everywhere else (`PCAReconstructor`/`Autoencoder`/`LSTMForecaster`
       all calibrate from a *separate* held-out nominal set). `calibrate_threshold()`
       here fixes `median_`/`iqr_` from validation data instead, same as those.
    2. **The learned graph is frozen after `fit()`, not after `partial_fit()`.** Within
       `fit()` itself, the graph co-evolves with the sensor embeddings (rebuilt from
       them at the end of every epoch, see `fit()`'s own docstring) so it reflects
       something the model actually learned, not its random initialization. But the
       paper never addresses online use (its own "Future work" section lists "online
       training methods" as unaddressed) -- `partial_fit()` continues training every
       OTHER weight (sensor embeddings included) but never re-derives the top-k
       adjacency mask again after `fit()` converged it, so a single later fold's data
       can't reshape which channels are considered related to which. This is a real,
       explicit design choice: letting the graph itself keep changing every fold risks
       unstable attention weights (and therefore unstable scores/explanations)
       fold-to-fold, at the cost of never adapting the topology to a channel relationship
       that genuinely changes later in the mission (e.g. a failing sensor decoupling from
       what it used to correlate with).
    """

    def __init__(
        self,
        n_channels: int,
        window_length: int = 10,
        embed_dim: int = 32,
        top_k: int = 15,
        hidden_size: int = 64,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.n_channels = n_channels
        self.window_length = window_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.net = _GDNNet(n_channels, window_length, embed_dim, hidden_size).to(device)
        self.graph_built = False
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_

    def _ensure_standardization_stats(self, values: np.ndarray) -> None:
        """Sets `mean_`/`std_` once, from the FIRST batch ever seen (by `fit()` or
        `partial_fit()`) -- never recomputed afterwards, same reasoning as
        `PCAReconstructor.partial_fit`: recomputing from just the newest batch would
        silently redefine "normal" every call.
        """
        if self.mean_ is None:
            self.mean_ = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            self.std_ = std

    def fit(self, values: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, stride: int = 1, verbose: bool = True) -> "GDN":
        """Builds an initial graph from the (still random) sensor embeddings so the
        first epoch's forward pass has *some* adjacency to mask attention with, then
        REBUILDS it from the current embeddings at the end of every epoch during
        training (`rebuild_graph_each_epoch=True` below), so the final graph reflects
        what the model actually learned rather than its random initialization. Call
        `calibrate_threshold()` afterwards, same as `PCAReconstructor`.
        """
        self._ensure_standardization_stats(values)
        with torch.no_grad():
            self.net.adjacency.copy_(_build_graph(self.net.sensor_embeddings.detach(), self.top_k))
        self.graph_built = True
        self._train(values, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose, rebuild_graph_each_epoch=True)
        return self

    def partial_fit(self, values: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3, stride: int = 1, verbose: bool = True) -> "GDN":
        """Continue training on new nominal data -- builds the graph on the FIRST call
        only (self-initializing, same pattern as `PCAReconstructor.partial_fit`), then
        never touches `adjacency` again (see class docstring, deviation 2) -- unlike
        `fit()`, which now actively co-evolves the graph with the embeddings throughout
        training, `partial_fit()` deliberately keeps whatever graph `fit()` converged to
        fixed, for online-stability reasons already documented there.
        """
        self._ensure_standardization_stats(values)
        if not self.graph_built:
            with torch.no_grad():
                self.net.adjacency.copy_(_build_graph(self.net.sensor_embeddings.detach(), self.top_k))
            self.graph_built = True
        self._train(values, epochs=epochs, batch_size=batch_size, lr=lr, stride=stride, verbose=verbose, rebuild_graph_each_epoch=False)
        return self

    def _train(
        self, values: np.ndarray, epochs: int, batch_size: int, lr: float, stride: int, verbose: bool, rebuild_graph_each_epoch: bool = False
    ) -> None:
        values = self._standardize(np.asarray(values, dtype=np.float64))
        X, y = build_forecast_windows(values, self.window_length, horizon=1, stride=stride)
        if X.shape[0] == 0:
            raise ValueError(f"Not enough rows ({values.shape[0]}) to build a single window of window_length={self.window_length}")
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y[:, 0, :], dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n = X_t.shape[0]
        self.net.train()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for epoch in range(epochs):
            perm = torch.randperm(n, generator=generator)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                pred, _ = self.net(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.shape[0]
            if rebuild_graph_each_epoch:
                # Co-evolve the graph with the embeddings that define it, instead of
                # leaving it fixed at whatever (initially random) state it was built from.
                with torch.no_grad():
                    self.net.adjacency.copy_(_build_graph(self.net.sensor_embeddings.detach(), self.top_k))
            if verbose:
                print(f"[GDN] epoch {epoch + 1}/{epochs} loss={total_loss / n:.5f}")

    def _channel_errors(self, values: np.ndarray, chunk_rows: int = 20_000) -> np.ndarray:
        """Per-row, per-channel absolute forecast error -- `Err_i(t)` in the paper's
        eq. 11. Chunked with carried-over context, mirroring `LSTMForecaster._residuals`
        (scoring a multi-million-row split as one `build_forecast_windows` call would
        try to materialize all its overlapping windows in memory at once).
        """
        self.net.eval()
        values = self._standardize(np.asarray(values, dtype=np.float64))
        n = values.shape[0]
        errors = np.zeros((n, self.n_channels))
        if n <= self.window_length:
            return errors
        chunk_start = 0
        with torch.no_grad():
            while chunk_start < n:
                chunk_end = min(chunk_start + chunk_rows, n)
                ctx_start = max(0, chunk_start - self.window_length)
                X, y = build_forecast_windows(values[ctx_start:chunk_end], self.window_length, horizon=1, stride=1)
                if X.shape[0] > 0:
                    X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
                    pred, _ = self.net(X_t)
                    err = np.abs(pred.cpu().numpy() - y[:, 0, :])
                    out_start = ctx_start + self.window_length
                    errors[out_start : out_start + err.shape[0]] = err
                chunk_start = chunk_end
        return errors

    def calibrate_threshold(self, validation_values: np.ndarray, percentile: float = 99.0) -> "GDN":
        """Fixes `median_`/`iqr_` (eq. 12's per-channel robust normalization stats) from
        held-out nominal data, then a percentile-of-score threshold from the same data --
        see class docstring, deviation 1, for why this differs from the paper's own
        self-referential recomputation.
        """
        if self.mean_ is None:
            raise RuntimeError("GDN must be fit() or partial_fit() before calibrate_threshold()")
        errors = self._channel_errors(validation_values)
        self.median_ = np.median(errors, axis=0)
        iqr = np.percentile(errors, 75, axis=0) - np.percentile(errors, 25, axis=0)
        iqr[iqr == 0] = 1.0
        self.iqr_ = iqr
        self.threshold_ = float(np.percentile(self.score(validation_values), percentile))
        return self

    def score_per_channel(self, values: np.ndarray) -> np.ndarray:
        """`(n_rows, n_channels)` robust-normalized deviation per channel -- eq. 12's
        `a_i(t)` for every channel, not yet aggregated. The explainability primitive
        everything else here (`score`, `most_anomalous_channels`) is built on: lets a
        caller see WHICH channel(s) drove a given row's anomaly score, not just that the
        row was anomalous.
        """
        if self.median_ is None:
            raise RuntimeError("GDN must be calibrated (calibrate_threshold) before scoring")
        errors = self._channel_errors(values)
        return (errors - self.median_) / self.iqr_

    def score(self, values: np.ndarray) -> np.ndarray:
        """Eq. 13: `A(t) = max_i a_i(t)` -- the single worst-offending channel drives the
        row-level score (plausible, per the paper, since an anomaly may affect only a
        small subset of channels or even just one).
        """
        return self.score_per_channel(values).max(axis=1)

    def predict(self, values: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise RuntimeError("No threshold: pass one explicitly or call calibrate_threshold() first")
        return self.score(values) > thr

    def most_anomalous_channels(self, values: np.ndarray, channel_names: Optional[list[str]] = None, top_n: int = 3) -> list:
        """For each row, the `top_n` channels with the highest normalized deviation --
        e.g. for a flagged row, `most_anomalous_channels(row)[0]` names the channel(s)
        an operator should look at first, instead of just "something is wrong".
        """
        normalized = self.score_per_channel(values)
        top_n = min(top_n, normalized.shape[1])
        order = np.argsort(-normalized, axis=1)[:, :top_n]
        if channel_names is None:
            return order.tolist()
        return [[channel_names[i] for i in row] for row in order]

    def graph_edges(self, channel_names: Optional[list[str]] = None) -> list[tuple]:
        """The learned dependency graph as `(source, target)` pairs (excluding
        self-loops) -- which channels the model considers related to which, directly
        from the frozen `adjacency` mask. Independent of any particular row/window.
        """
        adjacency = self.net.adjacency.detach().cpu().numpy()
        edges = [(i, j) for i in range(self.n_channels) for j in range(self.n_channels) if adjacency[i, j] and i != j]
        if channel_names is None:
            return edges
        return [(channel_names[i], channel_names[j]) for i, j in edges]

    def attention_for_window(self, window: np.ndarray) -> np.ndarray:
        """`(n_channels, n_channels)` attention weights for ONE window (shape
        `(window_length, n_channels)`) -- `attention[i, j]` is how much channel i's
        forecast for this specific window relied on neighbor j, i.e. the explanation for
        one specific alert rather than the graph's general structure (`graph_edges`).
        """
        self.net.eval()
        x = torch.tensor(window[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, attention = self.net(x)
        return attention[0].cpu().numpy()
