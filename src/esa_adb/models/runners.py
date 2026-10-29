from __future__ import annotations

import numpy as np

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

from esa_adb.models.base import window_scores_to_row_scores
from esa_adb.models.heavy import AnomalyTransformerLite, Autoencoder, CRTFAutoencoder, CRTFFull, GDN, LOEAutoencoder, LSTMForecaster, MemStream, PLDAAutoencoder, RoCAFull, RoCALite
from esa_adb.models.refinement import USDRRefiner
from esa_adb.models.lightweight import (
    ADDAEILForest,
    GlobalSTD,
    HalfSpaceTrees,
    GBMForecaster,
    LaggedRidgeForecaster,
    MatrixProfileDetector,
    NearestNominalWindow,
    OnlineGlobalSTD,
    DynamicFactorModel,
    EchoStateForecaster,
    GaussianCopulaDetector,
    HotellingPCA,
    LodaDetector,
    VARMahalanobis,
    KalmanAdaptiveTrend,
    KalmanBlockMultivariate,
    KalmanLocalTrend,
    KalmanRobustTrend,
    TCopulaDetector,
    OnlineIsolationForest,
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

    def run_config(self) -> dict:
        """Model-specific parameters included in the append-only experiment record."""
        return {}


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


class KalmanRunner(ScoreRunner):
    name = "KalmanLocalTrend"

    def _new_model(self) -> KalmanLocalTrend:
        return KalmanLocalTrend()

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class KalmanBlockRunner(ScoreRunner):
    name = "KalmanBlockMultivariate"

    def __init__(self, channel_groups: dict):
        super().__init__()
        self.channel_groups = channel_groups

    def _new_model(self) -> KalmanBlockMultivariate:
        return KalmanBlockMultivariate(self.channel_groups)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class KalmanGlobalRunner(KalmanBlockRunner):
    """Same KalmanBlockMultivariate mechanism, but one global block covering
    every channel instead of per-subsystem blocks. Worth testing because plain
    PCA beats its own subsystem-split version, so the same question is worth
    asking here instead of assuming the subsystem grouping is best."""

    name = "KalmanGlobalMultivariate"

    def __init__(self, n_channels: int):
        super().__init__(channel_groups={"global": list(range(n_channels))})


class DynamicFactorRunner(ScoreRunner):
    name = "DynamicFactorModel"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = DynamicFactorModel().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = DynamicFactorModel()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class GaussianCopulaRunner(ScoreRunner):
    name = "GaussianCopulaDetector"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = GaussianCopulaDetector().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = GaussianCopulaDetector()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class ADDAEILRunner(ScoreRunner):
    name = "ADDAEILForest"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = ADDAEILForest().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = ADDAEILForest()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class KalmanRobustRunner(ScoreRunner):
    name = "KalmanRobustTrend"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = KalmanRobustTrend().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = KalmanRobustTrend()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class KalmanAdaptiveRunner(ScoreRunner):
    name = "KalmanAdaptiveTrend"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = KalmanAdaptiveTrend().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = KalmanAdaptiveTrend()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


class TCopulaRunner(ScoreRunner):
    name = "TCopulaDetector"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = TCopulaDetector().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = TCopulaDetector()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


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


class USDRSubsystemRunner(SubsystemRunner):
    """Online subsystem PCA preceded by fold-local, label-free USDR refinement."""

    name = "USDR-SubsystemPCAEnsemble"

    def __init__(self, channel_groups: dict, seed: int = 42):
        super().__init__(channel_groups)
        self.refiner = USDRRefiner(seed=seed)
        self.refinement_history = []

    def _new_model(self):
        return SubsystemPCAEnsemble(self.channel_groups, min_channels=3, n_components=0.95)

    def _refine(self, values):
        retained, diagnostics = self.refiner.select(values, self._new_model, self._scalar_scores)
        self.refinement_history.append(
            {
                "input_rows": diagnostics.input_rows,
                "retained_rows": diagnostics.retained_rows,
                "rejected_rows": diagnostics.rejected_rows,
                "score_threshold": diagnostics.score_threshold,
            }
        )
        return retained

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model().fit(self._refine(fit_values))
        return self._scalar_scores(model, calib_values), self._scalar_scores(model, eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(self._refine(fit_values))

    def run_config(self) -> dict:
        return {
            "usdr_n_models": self.refiner.n_models,
            "usdr_memberships_per_row": self.refiner.memberships_per_row,
            "usdr_mad_multiplier": self.refiner.mad_multiplier,
            "usdr_max_reject_fraction": self.refiner.max_reject_fraction,
            "usdr_refinement_history": self.refinement_history,
        }


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


class OnlineIForestRunner(ScoreRunner):
    name = "OnlineIsolationForest"

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed

    def _new_model(self):
        return OnlineIsolationForest(n_trees=32, omega=2048, eta=32, stride=10, seed=self.seed)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"oif_n_trees": 32, "oif_omega": 2048, "oif_eta": 32, "oif_stride": 10, "seed": self.seed}


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


class GBMForecasterRunner(ScoreRunner):
    name = "GBMForecaster"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = GBMForecaster().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = GBMForecaster()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)

    def score_rows_per_channel(self, values):
        return self.online_model.score_per_channel(values)


class MatrixProfileRunner(ScoreRunner):
    """No persistent online state. There is no incremental multivariate Matrix
    Profile algorithm, so `update()` just replaces the reference window each fold."""

    name = "MatrixProfileDetector"

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = MatrixProfileDetector().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = MatrixProfileDetector()
        self.online_model.fit(fit_values)

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

    def run_config(self) -> dict:
        return {
            "hidden_sizes": [32, 16],
            "latent_size": 8,
            "refit_epochs": self.refit_epochs,
            "online_epochs": self.online_epochs,
            "online_lr": self.online_lr,
            "replay_rows": self.replay_rows,
            "seed": self.seed,
        }


class LOEAutoencoderRunner(ScoreRunner):
    """A matched Autoencoder baseline with LOE-H or LOE-S as the only difference."""

    def __init__(self, n_channels: int, assignment: str, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.assignment = assignment
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10
        self.name = f"LOEAutoencoder-{assignment[0].upper()}"

    def _new_model(self) -> LOEAutoencoder:
        return LOEAutoencoder(
            n_channels=self.n_channels,
            hidden_sizes=(32, 16),
            latent_size=8,
            assignment=self.assignment,
            contamination=0.10,
            margin=0.50,
            warmup_epochs=3,
            seed=self.seed,
        )

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=1024, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {
            "loe_assignment": self.assignment,
            "loe_contamination": 0.10,
            "loe_margin": 0.50,
            "loe_warmup_epochs": 3,
            "loe_refit_epochs": self.refit_epochs,
            "loe_online_epochs": self.online_epochs,
            "seed": self.seed,
        }


class CRTFAutoencoderRunner(ScoreRunner):
    """CLEANet-inspired contamination-resilient Autoencoder (see `crtf_autoencoder.py`)."""

    def __init__(self, n_channels: int, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10
        self.name = "CRTFAutoencoder"

    def _new_model(self) -> CRTFAutoencoder:
        return CRTFAutoencoder(
            n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8,
            n_clusters=8, contrastive_weight=0.5, weight_temperature=1.0, warmup_epochs=3,
            seed=self.seed,
        )

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=1024, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {
            "crtf_n_clusters": 8,
            "crtf_contrastive_weight": 0.5,
            "crtf_weight_temperature": 1.0,
            "crtf_warmup_epochs": 3,
            "crtf_refit_epochs": self.refit_epochs,
            "crtf_online_epochs": self.online_epochs,
            "seed": self.seed,
        }


class PLDAAutoencoderRunner(ScoreRunner):
    """PLDA-lite parameter-behavior weighting over the autoencoder (see `plda_autoencoder.py`)."""

    def __init__(self, n_channels: int, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10
        self.name = "PLDAAutoencoder"

    def _new_model(self) -> PLDAAutoencoder:
        return PLDAAutoencoder(n_channels=self.n_channels, ac_penalty=0.9, hn_boost=0.5, warmup_epochs=3, latent_size=8, seed=self.seed)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=1024, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"plda_ac_penalty": 0.9, "plda_hn_boost": 0.5, "plda_warmup_epochs": 3, "plda_refit_epochs": self.refit_epochs, "plda_online_epochs": self.online_epochs, "seed": self.seed}


class RoCALiteRunner(ScoreRunner):
    """RoCA-lite three-term objective with a fixed contamination fraction nu (see `roca_lite.py`)."""

    def __init__(self, n_channels: int, nu: float, name: str, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.nu = nu
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10
        self.name = name

    def _new_model(self) -> RoCALite:
        return RoCALite(n_channels=self.n_channels, nu=self.nu, seed=self.seed)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=1024, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=1024, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"roca_nu": self.nu, "roca_refit_epochs": self.refit_epochs, "roca_online_epochs": self.online_epochs, "seed": self.seed}


class AutoencoderMultiSeedRunner(ScoreRunner):
    """Same seed-averaging ensemble as CRTFMultiSeedRunner, for the blind
    Autoencoder baseline, whose single-seed score is known to swing over 15x
    by seed."""

    name = "Autoencoder-multiseed"

    def __init__(self, n_channels: int, seeds: tuple[int, ...] = (7, 42, 123)):
        super().__init__()
        self.seeds = seeds
        self.members = [AutoencoderRunner(n_channels, seed=s) for s in seeds]

    def refit_scores(self, fit_values, calib_values, eval_values):
        calib_scores, eval_scores = [], []
        for m in self.members:
            c, e = m.refit_scores(fit_values, calib_values, eval_values)
            calib_scores.append(c)
            eval_scores.append(e)
        return np.mean(calib_scores, axis=0), np.mean(eval_scores, axis=0)

    def update(self, fit_values):
        for m in self.members:
            m.update(fit_values)

    def score_rows(self, values):
        return np.mean([m.score_rows(values) for m in self.members], axis=0)

    def run_config(self) -> dict:
        return {"ensemble_seeds": list(self.seeds)}


class CRTFMultiSeedRunner(ScoreRunner):
    """Averages CRTFAutoencoder's scores across several independently-trained
    seeds before calibration, to reduce the seed variance PyTorch-backed
    models in this catalog can show."""

    name = "CRTFAutoencoder-multiseed"

    def __init__(self, n_channels: int, seeds: tuple[int, ...] = (7, 42, 123)):
        super().__init__()
        self.seeds = seeds
        self.members = [CRTFAutoencoderRunner(n_channels, seed=s) for s in seeds]

    def refit_scores(self, fit_values, calib_values, eval_values):
        calib_scores, eval_scores = [], []
        for m in self.members:
            c, e = m.refit_scores(fit_values, calib_values, eval_values)
            calib_scores.append(c)
            eval_scores.append(e)
        return np.mean(calib_scores, axis=0), np.mean(eval_scores, axis=0)

    def update(self, fit_values):
        for m in self.members:
            m.update(fit_values)

    def score_rows(self, values):
        return np.mean([m.score_rows(values) for m in self.members], axis=0)

    def run_config(self) -> dict:
        return {"ensemble_seeds": list(self.seeds), "member": self.members[0].run_config()}


class RoCAMultiSeedRunner(ScoreRunner):
    """Same seed-averaging ensemble as CRTFMultiSeedRunner, for RoCALite."""

    name = "RoCALite-multiseed"

    def __init__(self, n_channels: int, nu: float = 0.07, seeds: tuple[int, ...] = (7, 42, 123)):
        super().__init__()
        self.seeds = seeds
        self.members = [RoCALiteRunner(n_channels, nu=nu, name=f"RoCALite-seed{s}", seed=s) for s in seeds]

    def refit_scores(self, fit_values, calib_values, eval_values):
        calib_scores, eval_scores = [], []
        for m in self.members:
            c, e = m.refit_scores(fit_values, calib_values, eval_values)
            calib_scores.append(c)
            eval_scores.append(e)
        return np.mean(calib_scores, axis=0), np.mean(eval_scores, axis=0)

    def update(self, fit_values):
        for m in self.members:
            m.update(fit_values)

    def score_rows(self, values):
        return np.mean([m.score_rows(values) for m in self.members], axis=0)

    def run_config(self) -> dict:
        return {"ensemble_seeds": list(self.seeds), "member": self.members[0].run_config()}


class GDNRunner(ScoreRunner):
    """Wires GDN (graph-attention forecaster, see heavy/gdn.py) into the L2
    walk-forward protocol. online_scores is overridden because GDN needs
    calibrate_threshold() to set its own per-channel normalization before
    score() works at all. Its internal threshold is set but unused, since the
    external budget calibration picks the operating point instead."""

    name = "GDN"

    def __init__(self, n_channels: int, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.seed = seed

    def _new_model(self) -> GDN:
        # Scaled down from the paper's defaults, since this trains on CPU
        # across many folds instead of once offline.
        return GDN(n_channels=self.n_channels, window_length=10, embed_dim=16, top_k=10, hidden_size=32, seed=self.seed)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=10, batch_size=256, stride=20, verbose=False)
        model.calibrate_threshold(calib_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values, epochs=5, batch_size=256, stride=20, verbose=False)

    def online_scores(self, fit_values, calib_values, eval_values):
        self.update(fit_values)
        self.online_model.calibrate_threshold(calib_values)
        return self.online_model.score(calib_values), self.online_model.score(eval_values)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"gdn_window_length": 10, "gdn_embed_dim": 16, "gdn_top_k": 10, "gdn_hidden_size": 32, "seed": self.seed}


class FusionRunner(ScoreRunner):
    """Rank-normalizes each member's row scores within its own fold-local array (not a
    raw average, since the members' score scales aren't comparable: ECOD's copula score,
    an LSTM forecast residual, and an isolation-forest depth live on different scales),
    then averages the ranks into one fused score. Unlike the OR-of-alarms ensemble this
    project already tried (`scripts/run_l2_ensemble_overlap.py`), this combines the
    underlying continuous scores BEFORE a single threshold is calibrated, so it doesn't
    pay for each member's own alarm budget separately."""

    name = "HeterogeneousFusion"

    def __init__(self, members: dict[str, ScoreRunner]):
        super().__init__()
        self.members = members

    @staticmethod
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        return order / max(x.shape[0] - 1, 1)

    def refit_scores(self, fit_values, calib_values, eval_values):
        calib_ranks, eval_ranks = [], []
        for m in self.members.values():
            c, e = m.refit_scores(fit_values, calib_values, eval_values)
            calib_ranks.append(self._rank(c))
            eval_ranks.append(self._rank(e))
        return np.mean(calib_ranks, axis=0), np.mean(eval_ranks, axis=0)

    def update(self, fit_values):
        for m in self.members.values():
            m.update(fit_values)

    def score_rows(self, values):
        ranks = [self._rank(m.score_rows(values)) for m in self.members.values()]
        return np.mean(ranks, axis=0)

    def run_config(self) -> dict:
        return {"fusion_members": list(self.members)}


class CRTFFullRunner(ScoreRunner):
    """Faithful CRTF, CLEANet's full method (see crtf_full.py), the follow-up
    to CRTFAutoencoderRunner once the paper's complete text was available."""

    def __init__(self, n_channels: int, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10
        self.name = "CRTFFull"

    def _new_model(self) -> CRTFFull:
        return CRTFFull(
            n_channels=self.n_channels, window_length=20, hidden_time=8, hidden_feat=8,
            n_layers=2, k_neighbors=10, alpha=5.0, lam=0.5, temperature=0.5, min_cluster_size=5,
            seed=self.seed,
        )

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=256, stride=5, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=256, stride=5, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=256, stride=5, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"crtf_full_window_length": 20, "crtf_full_k": 10, "crtf_full_alpha": 5.0, "crtf_full_lambda": 0.5, "seed": self.seed}


class RoCAFullRunner(ScoreRunner):
    """Faithful RoCA, TCN plus Seq2Seq LSTM (see roca_full.py), the follow-up
    to RoCALiteRunner once the paper's full text and architecture were read."""

    def __init__(self, n_channels: int, nu: float, name: str, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.nu = nu
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10
        self.name = name

    def _new_model(self) -> RoCAFull:
        return RoCAFull(n_channels=self.n_channels, window_length=24, tcn_hidden=16, lstm_hidden=16, proj_size=16, nu=self.nu, seed=self.seed)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=256, stride=5, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=256, stride=5, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=256, stride=5, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"roca_full_nu": self.nu, "roca_full_window_length": 24, "roca_full_refit_epochs": self.refit_epochs, "seed": self.seed}


class AnomalyTransformerLiteRunner(ScoreRunner):
    """Anomaly Transformer (Xu et al., ICLR 2022) scaled down for CPU, see
    anomaly_transformer_lite.py. Scores association discrepancy instead of
    reconstruction residual, a different signal from the other models here."""

    name = "AnomalyTransformerLite"

    def __init__(self, n_channels: int, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.seed = seed
        self.refit_epochs = 20
        self.online_epochs = 10

    def _new_model(self) -> AnomalyTransformerLite:
        return AnomalyTransformerLite(n_channels=self.n_channels, window_length=30, d_model=32, n_heads=4, n_layers=2, ff_hidden=64, seed=self.seed)

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.refit_epochs, batch_size=256, stride=5, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.online_epochs, batch_size=256, stride=5, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, epochs=self.online_epochs, batch_size=256, stride=5, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {"at_window_length": 30, "at_d_model": 32, "at_n_layers": 2, "seed": self.seed}


class MemStreamRunner(ScoreRunner):
    """MemStream: encoder pretrained once, then a frozen-weight bounded memory bank
    absorbs drift online with no further gradient descent (see `memstream.py`)."""

    def __init__(self, n_channels: int, mem_size: int = 2048, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.mem_size = mem_size
        self.seed = seed
        self.pretrain_epochs = 20
        self.name = "MemStream"

    def _new_model(self) -> MemStream:
        return MemStream(
            n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8,
            mem_size=self.mem_size, seed=self.seed,
        )

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self._new_model()
        model.fit(fit_values, epochs=self.pretrain_epochs, batch_size=1024, verbose=False)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self._new_model()
            self.online_model.fit(fit_values, epochs=self.pretrain_epochs, batch_size=1024, verbose=False)
        else:
            self.online_model.partial_fit(fit_values, verbose=False)

    def score_rows(self, values):
        return self.online_model.score(values)

    def run_config(self) -> dict:
        return {
            "mem_size": self.mem_size,
            "latent_size": 8,
            "hidden_sizes": [32, 16],
            "pretrain_epochs": self.pretrain_epochs,
            "seed": self.seed,
        }


class LSTMRunner(ScoreRunner):
    name = "LSTMForecaster"

    def __init__(self, n_channels: int, seed: int = 42):
        super().__init__()
        self.n_channels = n_channels
        self.seed = seed

    def _new_model(self):
        # Deliberately downsized vs. LSTMForecaster defaults, same rationale as
        # demo_full_mission_walk_forward.py: ~20 folds of real CPU training per variant.
        return LSTMForecaster(n_channels=self.n_channels, input_length=50, hidden_size=32, num_layers=1, seed=self.seed)

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


ROCA_NU = {"RoCALite": 0.07, "RoCALite-nu05": 0.05, "RoCALite-nu10": 0.10}

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


class FactoryRunner(ScoreRunner):
    """Generic runner for a detector exposing `fit`/`partial_fit`/`score` (and nothing
    model-specific): only the factory differs, same idea as `PyODRunner`."""

    def __init__(self, name: str, factory):
        super().__init__()
        self.name = name
        self.factory = factory

    def refit_scores(self, fit_values, calib_values, eval_values):
        model = self.factory().fit(fit_values)
        return model.score(calib_values), model.score(eval_values)

    def update(self, fit_values):
        if self.online_model is None:
            self.online_model = self.factory()
        self.online_model.partial_fit(fit_values)

    def score_rows(self, values):
        return self.online_model.score(values)


def _chronos():
    from esa_adb.models.lightweight.chronos_residual import ChronosResidualDetector

    return ChronosResidualDetector()


FACTORY_MODELS = {
    "ChronosResidual": _chronos,
    "VARMahalanobis": VARMahalanobis,
    "HotellingPCA": HotellingPCA,
    "LodaDetector": LodaDetector,
    "EchoStateForecaster": EchoStateForecaster,
}


def cusum(z: np.ndarray, k: float) -> np.ndarray:
    """One-sided CUSUM S_t = max(0, S_{t-1} + z_t - k)."""
    out = np.empty_like(z)
    s = 0.0
    for t in range(z.shape[0]):
        s = max(0.0, s + z[t] - k)
        out[t] = s
    return out


class CUSUMRunner(ScoreRunner):
    """Wraps another runner and replaces its score by a CUSUM of the robustly
    standardized score (median/MAD, z capped at `z_cap` so one huge spike cannot keep the
    statistic alive for thousands of rows).

    A per-row score only reacts to a single large deviation; CUSUM also accumulates a
    persistent small shift, the signature of a slow degradation or a mode change that no
    single row makes alarming. Restarts from zero at every chunk (calibration and
    evaluation are separate contiguous segments).

    The median/MAD reference comes from the previous OUT-OF-SAMPLE chunk this runner
    scored (calibration -> evaluation -> next fold's calibration ...), never from the fit
    rows: a model scores its own training rows lower than unseen ones, so an in-sample
    reference sits below every later chunk, `z - k` is positive on average, and the CUSUM
    ramps without bound into one month-long alarm (that was a real bug in the first
    version of this wrapper). The very first chunk is standardized by itself."""

    def __init__(self, base: ScoreRunner, k: float = 0.5, z_cap: float = 10.0):
        super().__init__()
        self.base, self.k, self.z_cap = base, k, z_cap
        self.name = f"CUSUM-{base.name}"
        self.ref: tuple[float, float] | None = None

    @staticmethod
    def _robust_ref(scores: np.ndarray) -> tuple[float, float]:
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med))) * 1.4826
        return med, mad if mad > 0 else float(scores.std() or 1.0)

    def _transform(self, scores: np.ndarray, ref: tuple[float, float]) -> np.ndarray:
        return cusum(np.minimum((scores - ref[0]) / ref[1], self.z_cap), self.k)

    def refit_scores(self, fit_values, calib_values, eval_values):
        calib_s, eval_s = self.base.refit_scores(fit_values, calib_values, eval_values)
        calib_ref = self._robust_ref(calib_s)
        return self._transform(calib_s, calib_ref), self._transform(eval_s, calib_ref)

    def update(self, fit_values):
        self.base.update(fit_values)

    def score_rows(self, values):
        s = self.base.score_rows(values)
        new_ref = self._robust_ref(s)
        out = self._transform(s, self.ref if self.ref is not None else new_ref)
        self.ref = new_ref
        return out

    def run_config(self) -> dict:
        return {"cusum_k": self.k, "cusum_z_cap": self.z_cap, "cusum_base": self.base.name, "cusum_ref": "previous out-of-sample chunk"}


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
        elif name == "KalmanLocalTrend":
            runners[name] = KalmanRunner()
        elif name == "KalmanBlockMultivariate":
            runners[name] = KalmanBlockRunner(channel_groups)
        elif name == "KalmanGlobalMultivariate":
            runners[name] = KalmanGlobalRunner(n_channels)
        elif name == "DynamicFactorModel":
            runners[name] = DynamicFactorRunner()
        elif name == "GaussianCopulaDetector":
            runners[name] = GaussianCopulaRunner()
        elif name == "TCopulaDetector":
            runners[name] = TCopulaRunner()
        elif name == "KalmanRobustTrend":
            runners[name] = KalmanRobustRunner()
        elif name == "KalmanAdaptiveTrend":
            runners[name] = KalmanAdaptiveRunner()
        elif name == "PCAReconstructor":
            runners[name] = PCARunner()
        elif name == "SubsystemPCAEnsemble":
            if channel_groups:
                runners[name] = SubsystemRunner(channel_groups)
        elif name == "USDR-SubsystemPCAEnsemble":
            if channel_groups:
                runners[name] = USDRSubsystemRunner(channel_groups, seed=ae_seed)
        elif name == "WindowedIsolationForest":
            runners[name] = WindowedIForestRunner(spec)
        elif name == "OnlineIsolationForest":
            runners[name] = OnlineIForestRunner(seed=ae_seed)
        elif name == "ADDAEILForest":
            runners[name] = ADDAEILRunner()
        elif name == "HalfSpaceTrees":
            runners[name] = HalfSpaceTreesRunner()
        elif name == "LaggedRidgeForecaster":
            runners[name] = LaggedRidgeRunner()
        elif name == "GBMForecaster":
            runners[name] = GBMForecasterRunner()
        elif name == "MatrixProfileDetector":
            runners[name] = MatrixProfileRunner()
        elif name == "NearestNominalWindow":
            runners[name] = NearestNominalWindowRunner(spec)
        elif name == "Autoencoder":
            runners[name] = AutoencoderRunner(n_channels, seed=ae_seed)
        elif name == "Autoencoder-replay":
            runners[name] = AutoencoderRunner(
                n_channels, replay_rows=ae_replay_rows or 50_000, online_epochs=ae_online_epochs, online_lr=ae_online_lr, seed=ae_seed
            )
        elif name == "LOEAutoencoder-H":
            runners[name] = LOEAutoencoderRunner(n_channels, assignment="hard", seed=ae_seed)
        elif name == "LOEAutoencoder-S":
            runners[name] = LOEAutoencoderRunner(n_channels, assignment="soft", seed=ae_seed)
        elif name == "MemStream":
            runners[name] = MemStreamRunner(n_channels, seed=ae_seed)
        elif name == "CRTFAutoencoder":
            runners[name] = CRTFAutoencoderRunner(n_channels, seed=ae_seed)
        elif name == "Autoencoder-multiseed":
            runners[name] = AutoencoderMultiSeedRunner(n_channels)
        elif name == "CRTFAutoencoder-multiseed":
            runners[name] = CRTFMultiSeedRunner(n_channels)
        elif name == "CRTFFull":
            runners[name] = CRTFFullRunner(n_channels, seed=ae_seed)
        elif name == "RoCAFull":
            runners[name] = RoCAFullRunner(n_channels, nu=0.07, name=name, seed=ae_seed)
        elif name == "AnomalyTransformerLite":
            runners[name] = AnomalyTransformerLiteRunner(n_channels, seed=ae_seed)
        elif name == "RoCALite-multiseed":
            runners[name] = RoCAMultiSeedRunner(n_channels, nu=0.07)
        elif name == "GDN":
            runners[name] = GDNRunner(n_channels, seed=ae_seed)
        elif name == "HeterogeneousFusion":
            fusion_members = ["PyOD-ECOD", "LSTMForecaster", "WindowedIsolationForest", "PyOD-COPOD", "PyOD-IForest"]
            runners[name] = FusionRunner(build_runners(fusion_members, n_channels, channel_groups, spec, ae_seed=ae_seed))
        elif name in FACTORY_MODELS:
            runners[name] = FactoryRunner(name, FACTORY_MODELS[name])
        elif name.startswith("CUSUM-"):
            base = name.removeprefix("CUSUM-")
            runners[name] = CUSUMRunner(build_runners([base], n_channels, channel_groups, spec, ae_replay_rows, ae_online_epochs, ae_online_lr, ae_seed)[base])
        elif name == "PLDAAutoencoder":
            runners[name] = PLDAAutoencoderRunner(n_channels, seed=ae_seed)
        elif name in ROCA_NU:
            runners[name] = RoCALiteRunner(n_channels, nu=ROCA_NU[name], name=name, seed=ae_seed)
        elif name == "LSTMForecaster":
            runners[name] = LSTMRunner(n_channels, seed=ae_seed)
        elif name in PYOD_FACTORIES:
            runners[name] = PyODRunner(name, PYOD_FACTORIES[name])
        else:
            raise ValueError(f"Unknown model name: {name}")
    return runners
