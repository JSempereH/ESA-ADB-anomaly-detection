from __future__ import annotations

import numpy as np

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

from esa_adb.models.base import window_scores_to_row_scores
from esa_adb.models.heavy import Autoencoder, LSTMForecaster
from esa_adb.models.lightweight import (
    GlobalSTD,
    HalfSpaceTrees,
    LaggedRidgeForecaster,
    NearestNominalWindow,
    OnlineGlobalSTD,
    PCAReconstructor,
    PyODDetector,
    WindowedIsolationForest,
)
from esa_adb.models.subsystem_ensemble import SubsystemPCAEnsemble
from esa_adb.preprocessing.windowing import WindowSpec


class ScoreRunner:
    """One instance per model type per benchmark run. All methods deal in RAW row-level
    anomaly scores -- thresholding, smoothing and alarm post-processing are the caller's
    job (`esa_adb.benchmark`), so every threshold/postprocess strategy sees identical
    model output and model code never needs re-running to compare strategies.

    Subclasses implement the online path as two primitives -- `update(fit_values)`
    (update ONE persisted instance, never reset) and `score_rows(values)` -- which the
    walk-forward engine composes via `online_scores()` and the adaptive-cadence engine
    (`benchmark.run_adaptive_cadence`) calls directly on its own schedule. `refit_scores`
    builds a brand-new model from only this fold's fit data. Window-level models
    broadcast their window scores back to rows internally, so the caller only ever sees
    row-level scores.
    """

    name = "base"

    def __init__(self):
        self.online_model = None

    def refit_scores(self, fit_values: np.ndarray, calib_values: np.ndarray, eval_values: np.ndarray):
        raise NotImplementedError

    def update(self, fit_values: np.ndarray) -> None:
        raise NotImplementedError

    def score_rows(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_rows_per_channel(self, values: np.ndarray) -> np.ndarray:
        """`(n_rows, n_channels)` per-channel scores from the persisted online model,
        for channel-attributed alarms (the paper's channel-aware evaluation). Only
        implemented by runners whose model exposes `score_per_channel`."""
        raise NotImplementedError(f"{self.name} has no per-channel score")

    def online_scores(self, fit_values: np.ndarray, calib_values: np.ndarray, eval_values: np.ndarray):
        self.update(fit_values)
        return self.score_rows(calib_values), self.score_rows(eval_values)


class GlobalSTDRunner(ScoreRunner):
    name = "GlobalSTD"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = GlobalSTD().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = OnlineGlobalSTD(n_channels=fit_values.shape[1])
        self.online_model.fit(fit_values)  # Welford's algorithm -- accumulates, never resets

    def score_rows(self, values):
        return self.online_model.score(values)

    def score_rows_per_channel(self, values):
        return self.online_model.score_per_channel(values)


class PCARunner(ScoreRunner):
    name = "PCAReconstructor"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = PCAReconstructor(n_components=0.95).fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = PCAReconstructor(n_components=0.95)
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)

    def score_rows_per_channel(self, values):
        return self.online_model.score_per_channel(values)


class SubsystemRunner(ScoreRunner):
    """Scalarized as the max standardized reconstruction error across subsystems
    ("how anomalous is the worst-offending subsystem") so one threshold strategy
    applies uniformly -- same simplification demo_dynamic_threshold.py already made.
    """

    name = "SubsystemPCAEnsemble"

    def __init__(self, channel_groups: dict):
        super().__init__()
        self.channel_groups = channel_groups

    def _scalar_scores(self, model, values):
        return np.max(np.stack(list(model.score_per_subsystem(values).values())), axis=0)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = SubsystemPCAEnsemble(self.channel_groups, min_channels=3, n_components=0.95).fit(fit_values)
        return self._scalar_scores(model, calib_values), self._scalar_scores(model, eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = SubsystemPCAEnsemble(self.channel_groups, min_channels=3, n_components=0.95)
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self._scalar_scores(self.online_model, values)


def _window_to_row_scores(window_scores: np.ndarray, n_rows: int, spec: WindowSpec) -> np.ndarray:
    """Broadcast + patch: rows not covered by any window (a partial trailing block)
    inherit the series' minimum score instead of -inf, so downstream quantile
    thresholding never sees infinities.
    """
    row_scores = window_scores_to_row_scores(window_scores, n_rows, spec)
    uncovered = ~np.isfinite(row_scores)
    if uncovered.any():
        row_scores[uncovered] = window_scores.min() if window_scores.size else 0.0
    return row_scores


class WindowedIForestRunner(ScoreRunner):
    name = "WindowedIsolationForest"

    def __init__(self, spec: WindowSpec):
        super().__init__()
        self.spec = spec

    def _row_scores(self, model, values):
        return _window_to_row_scores(model.score(values), values.shape[0], self.spec)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = WindowedIsolationForest(self.spec, n_estimators=100, contamination=0.05).fit(fit_values)
        return self._row_scores(model, calib_values), self._row_scores(model, eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = WindowedIsolationForest(self.spec, n_estimators=100, contamination=0.05).fit(fit_values)
        else:
            self.online_model.partial_fit(fit_values, n_new_estimators=100)

    def score_rows(self, values):
        return self._row_scores(self.online_model, values)


class HalfSpaceTreesRunner(ScoreRunner):
    name = "HalfSpaceTrees"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = HalfSpaceTrees(n_channels=fit_values.shape[1]).fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = HalfSpaceTrees(n_channels=fit_values.shape[1]).fit(fit_values)
        else:
            self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class LaggedRidgeRunner(ScoreRunner):
    name = "LaggedRidgeForecaster"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = LaggedRidgeForecaster().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = LaggedRidgeForecaster()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)

    def score_rows_per_channel(self, values):
        return self.online_model.score_per_channel(values)


class NearestNominalWindowRunner(ScoreRunner):
    name = "NearestNominalWindow"

    def __init__(self, spec: WindowSpec):
        super().__init__()
        self.spec = spec

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = NearestNominalWindow(self.spec).fit(fit_values)
        return (
            _window_to_row_scores(model.score(calib_values), calib_values.shape[0], self.spec),
            _window_to_row_scores(model.score(eval_values), eval_values.shape[0], self.spec),
        )

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = NearestNominalWindow(self.spec)
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return _window_to_row_scores(self.online_model.score(values), values.shape[0], self.spec)


class AutoencoderRunner(ScoreRunner):
    """`replay_rows > 0` (the P1 "online replay" variant) keeps a reservoir-sampled
    buffer of past fit rows and mixes it into every `partial_fit`, so the online model
    rehearses old months while learning the new one (MemStream-style bounded memory,
    Bhatia et al. WWW 2022) instead of relying on SGD alone not to forget them.
    """

    name = "Autoencoder"

    def __init__(self, n_channels: int, refit_epochs: int = 20, online_epochs: int = 10, replay_rows: int = 0, online_lr: float = 1e-3, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.refit_epochs = refit_epochs
        self.online_epochs = online_epochs
        self.replay_rows = replay_rows
        self.online_lr = online_lr
        self.seed = seed
        self._replay_buffer: np.ndarray | None = None
        self._replay_seen = 0
        self._rng = np.random.default_rng(seed)
        if replay_rows > 0:
            self.name = "Autoencoder-replay"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = Autoencoder(n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8, seed=self.seed)
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=1024, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def _update_replay(self, fit_values: np.ndarray) -> None:
        """Reservoir sampling over every fit row ever seen: each maintains an equal
        probability of occupying the bounded buffer, so the rehearsal sample stays
        representative of the WHOLE mission history, not just recent months.
        """
        if self._replay_buffer is None:
            self._replay_buffer = fit_values[: self.replay_rows].copy()
            self._replay_seen = self._replay_buffer.shape[0]
            fit_values = fit_values[self.replay_rows :]
        for row in fit_values:
            self._replay_seen += 1
            if self._replay_buffer.shape[0] < self.replay_rows:
                self._replay_buffer = np.vstack([self._replay_buffer, row])
            else:
                j = self._rng.integers(0, self._replay_seen)
                if j < self.replay_rows:
                    self._replay_buffer[j] = row

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = Autoencoder(n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8, seed=self.seed)
        train_values = fit_values
        if self.replay_rows > 0 and self._replay_buffer is not None:
            train_values = np.vstack([fit_values, self._replay_buffer])
        self.online_model.partial_fit(train_values, epochs=self.online_epochs, batch_size=1024, lr=self.online_lr, verbose=False)
        if self.replay_rows > 0:
            self._update_replay(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class LSTMRunner(ScoreRunner):
    name = "LSTMForecaster"

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

    def _new_model(self):
        # Deliberately downsized vs. LSTMForecaster defaults, same rationale as
        # demo_full_mission_walk_forward.py: ~20 folds of real CPU training per variant.
        return LSTMForecaster(n_channels=self.n_channels, input_length=50, hidden_size=32, num_layers=1)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.partial_fit(fit_values, epochs=10, batch_size=256, stride=20, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values, epochs=5, batch_size=256, stride=20, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)


PYOD_FACTORIES = {
    "PyOD-ECOD": lambda: ECOD(),
    "PyOD-COPOD": lambda: COPOD(),
    "PyOD-IForest": lambda: IForest(random_state=42),
}


class PyODRunner(ScoreRunner):
    """One generic runner class for every PyOD-backed model in `PYOD_FACTORIES` --
    only the factory differs, so this covers ECOD/COPOD/IForest (and any future PyOD
    addition) without a new class per algorithm. See `PyODDetector` for how "online"
    fakes incremental learning via a reservoir-sampled buffer + full refit."""

    def __init__(self, name: str, factory, buffer_rows: int = 50_000):
        super().__init__()
        self.name = name
        self.factory = factory
        self.buffer_rows = buffer_rows

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = PyODDetector(self.factory).fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = PyODDetector(self.factory, buffer_rows=self.buffer_rows)
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


def build_runners(
    names: list[str],
    n_channels: int,
    channel_groups: dict,
    spec: WindowSpec,
    ae_replay_rows: int = 0,
    ae_online_epochs: int = 10,
    ae_online_lr: float = 1e-3,
    ae_seed: int = 42,
) -> dict[str, ScoreRunner]:
    runners: dict[str, ScoreRunner] = {}
    for name in names:
        if name == "GlobalSTD":
            runners[name] = GlobalSTDRunner()
        elif name == "PCAReconstructor":
            runners[name] = PCARunner()
        elif name == "SubsystemPCAEnsemble":
            if channel_groups:
                runners[name] = SubsystemRunner(channel_groups)
        elif name == "WindowedIsolationForest":
            runners[name] = WindowedIForestRunner(spec)
        elif name == "HalfSpaceTrees":
            runners[name] = HalfSpaceTreesRunner()
        elif name == "LaggedRidgeForecaster":
            runners[name] = LaggedRidgeRunner()
        elif name == "NearestNominalWindow":
            runners[name] = NearestNominalWindowRunner(spec)
        elif name == "Autoencoder":
            runners[name] = AutoencoderRunner(n_channels, seed=ae_seed)
        elif name == "Autoencoder-replay":
            runners[name] = AutoencoderRunner(
                n_channels, replay_rows=ae_replay_rows or 50_000, online_epochs=ae_online_epochs, online_lr=ae_online_lr, seed=ae_seed
            )
        elif name == "LSTMForecaster":
            runners[name] = LSTMRunner(n_channels)
        elif name in PYOD_FACTORIES:
            runners[name] = PyODRunner(name, PYOD_FACTORIES[name])
        else:
            raise ValueError(f"Unknown model name: {name}")
    return runners
