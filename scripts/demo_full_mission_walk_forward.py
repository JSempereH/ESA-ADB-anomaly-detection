#!/usr/bin/env python3
"""Walk-forward over the FULL Mission2 test period (no fold cap, ~20 monthly folds),
applying three training methodologies uniformly to every model used in the single-split
benchmark:

  - "refit": a brand-new model, trained from scratch EACH FOLD on only that fold's own
    month of data (like scripts/demo_walk_forward.py's "sliding, 1 month" variant).
  - "online": ONE model instance, built once and updated every fold via `partial_fit()`
    -- never refit, never discards earlier folds' influence on the model's weights.
  - "hybrid": AND-combining "refit"'s and "online"'s row-level predictions for that fold
    -- the same pairing PairedLearnerEnsemble does (see models/ensemble.py), computed
    directly here since the two predictions are already on hand.

Covers: GlobalSTD/OnlineGlobalSTD, PCAReconstructor, WindowedIsolationForest,
SubsystemPCAEnsemble, Autoencoder, LSTMForecaster. (HalfSpaceTrees already has its own
refit-vs-online comparison in scripts/demo_online_hst.py, extended to the full mission
alongside this script rather than duplicated here.)

Each fold's OWN training month is split 2/3 fit-sub-period / 1/3 calibration-sub-period
(same anti-"calibrate on training data" pattern used everywhere else in this project --
see demo_walk_forward.py's `_fit_pca` docstring for why).

Usage: python scripts/demo_full_mission_walk_forward.py --mission 2 [--max-folds N] [--models GlobalSTD,PCAReconstructor,...]
"""
import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.evaluation import find_best_threshold
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import broadcast_window_pred_to_rows, channel_matrix
from esa_adb.models.heavy import Autoencoder, LSTMForecaster
from esa_adb.models.lightweight import GlobalSTD, OnlineGlobalSTD, PCAReconstructor, WindowedIsolationForest
from esa_adb.models.subsystem_ensemble import SubsystemPCAEnsemble
from esa_adb.models.walk_forward import aggregate_fold_results, generate_walk_forward_folds
from esa_adb.preprocessing.labeling import filter_nominal_rows, row_is_non_nominal
from esa_adb.preprocessing.windowing import WindowSpec, label_windows, window_bounds_frame

ALL_MODELS = ["GlobalSTD", "PCAReconstructor", "WindowedIsolationForest", "SubsystemPCAEnsemble", "Autoencoder", "LSTMForecaster"]


def _load_slice(lf: pl.LazyFrame, start, end, target_channels: list[str]) -> pl.DataFrame:
    return lf.filter((pl.col("time") >= start) & (pl.col("time") < end)).select(["time", *target_channels]).sort("time").collect()


def _threshold(scores: np.ndarray, y_true: np.ndarray, beta: float = 0.5) -> float:
    if y_true.sum() > 0:
        tuned = find_best_threshold(scores, y_true, beta=beta)
        if tuned["f_beta"] > 0:
            return tuned["threshold"]
    return float(np.percentile(scores, 99.0))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta**2
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f_beta": f_beta, "beta": beta}


class ModelRunner:
    """One instance per model type. `refit()` builds a fresh model from just this fold's
    data; `online()` updates a single persisted instance across every fold. Both return
    row-level boolean predictions for the fold's eval period, so `hybrid = refit_pred &
    online_pred` can be computed directly by the caller without needing a shared object
    interface across such different underlying models.
    """

    name = "base"

    def __init__(self, target_channels, labels_long, config):
        self.target_channels = target_channels
        self.labels_long = labels_long
        self.config = config
        self.online_model = None

    def _calib_arrays(self, calib_df):
        return channel_matrix(calib_df, self.target_channels), row_is_non_nominal(calib_df, self.labels_long, self.target_channels)

    def refit(self, fit_values, calib_df, eval_values):
        raise NotImplementedError

    def online(self, fit_values, calib_df, eval_values):
        raise NotImplementedError

    def online_raw(self, fit_values, calib_df, eval_values):
        """Update the ONE persisted online model with this fold's data (same update
        `online()` does) and return `(calib_scores, calib_y_or_None, eval_scores,
        is_window_level)` instead of already-thresholded predictions -- lets a caller
        apply several different threshold *strategies* to the same trained model's
        scores without paying the training cost more than once. `calib_y` is `None` for
        models whose native calibration is percentile-only rather than label-tuned
        (matches `online()`'s own calibration choice for that model, so the "static"
        threshold strategy computed from these raw scores reproduces `online()`'s result
        exactly). Used by scripts/demo_dynamic_threshold.py; not used by this script's
        own `online()`, which keeps its original, independent implementation.
        """
        raise NotImplementedError


class GlobalSTDRunner(ModelRunner):
    name = "GlobalSTD"

    def refit(self, fit_values, calib_df, eval_values):
        model = GlobalSTD().fit(fit_values)
        calib_values, calib_y = self._calib_arrays(calib_df)
        thr = _threshold(model.score(calib_values), calib_y)
        return model.predict(eval_values, threshold=thr)

    def online(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = OnlineGlobalSTD(n_channels=fit_values.shape[1])
        self.online_model.fit(fit_values)  # Welford's algorithm -- accumulates, never resets
        calib_values, calib_y = self._calib_arrays(calib_df)
        thr = _threshold(self.online_model.score(calib_values), calib_y)
        return self.online_model.predict(eval_values, threshold=thr)

    def online_raw(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = OnlineGlobalSTD(n_channels=fit_values.shape[1])
        self.online_model.fit(fit_values)
        calib_values, calib_y = self._calib_arrays(calib_df)
        return self.online_model.score(calib_values), calib_y, self.online_model.score(eval_values), False


class PCARunner(ModelRunner):
    name = "PCAReconstructor"

    def refit(self, fit_values, calib_df, eval_values):
        model = PCAReconstructor(n_components=0.95).fit(fit_values)
        calib_values, calib_y = self._calib_arrays(calib_df)
        thr = _threshold(model.score(calib_values), calib_y)
        return model.predict(eval_values, threshold=thr)

    def online(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = PCAReconstructor(n_components=0.95)
        self.online_model.partial_fit(fit_values)
        calib_values, calib_y = self._calib_arrays(calib_df)
        thr = _threshold(self.online_model.score(calib_values), calib_y)
        return self.online_model.predict(eval_values, threshold=thr)

    def online_raw(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = PCAReconstructor(n_components=0.95)
        self.online_model.partial_fit(fit_values)
        calib_values, calib_y = self._calib_arrays(calib_df)
        return self.online_model.score(calib_values), calib_y, self.online_model.score(eval_values), False


class SubsystemRunner(ModelRunner):
    name = "SubsystemPCAEnsemble"

    def __init__(self, target_channels, labels_long, config, channel_groups):
        super().__init__(target_channels, labels_long, config)
        self.channel_groups = channel_groups

    def refit(self, fit_values, calib_df, eval_values):
        model = SubsystemPCAEnsemble(self.channel_groups, min_channels=3, n_components=0.95).fit(fit_values)
        calib_values = channel_matrix(calib_df, self.target_channels)
        model.calibrate_threshold(calib_values, percentile=99.0)
        return model.predict(eval_values)

    def online(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = SubsystemPCAEnsemble(self.channel_groups, min_channels=3, n_components=0.95)
        self.online_model.partial_fit(fit_values)
        calib_values = channel_matrix(calib_df, self.target_channels)
        self.online_model.calibrate_threshold(calib_values, percentile=99.0)
        return self.online_model.predict(eval_values)

    def online_raw(self, fit_values, calib_df, eval_values):
        # SubsystemPCAEnsemble has no single native score() (it manages one PCA per
        # subsystem, each with its own threshold) -- scalarized here as the max
        # standardized reconstruction error across subsystems ("how anomalous is the
        # worst-offending subsystem"), so ONE dynamic threshold strategy can be applied
        # uniformly, same as every other model. This is a simplification specific to
        # this comparison, not a replacement for the ensemble's native per-subsystem
        # OR/min-agreeing logic used by refit()/online() above.
        if self.online_model is None:
            self.online_model = SubsystemPCAEnsemble(self.channel_groups, min_channels=3, n_components=0.95)
        self.online_model.partial_fit(fit_values)
        calib_values, calib_y = self._calib_arrays(calib_df)
        calib_scores = np.max(np.stack(list(self.online_model.score_per_subsystem(calib_values).values())), axis=0)
        eval_scores = np.max(np.stack(list(self.online_model.score_per_subsystem(eval_values).values())), axis=0)
        return calib_scores, calib_y, eval_scores, False


class WindowedIForestRunner(ModelRunner):
    name = "WindowedIsolationForest"

    def __init__(self, target_channels, labels_long, config, spec: WindowSpec):
        super().__init__(target_channels, labels_long, config)
        self.spec = spec

    def _calib_window_scores(self, model, calib_df):
        calib_values = channel_matrix(calib_df, self.target_channels)
        calib_grid_end = calib_df["time"].max() + timedelta(seconds=self.config.target_frequency_seconds)
        calib_bounds = window_bounds_frame(calib_df["time"].min(), calib_grid_end, self.config.target_frequency_seconds, self.spec)
        calib_labelled = label_windows(
            self.labels_long, calib_bounds, calib_df["time"].min(), self.config.target_frequency_seconds, self.spec, channels=self.target_channels
        )
        calib_window_y = (~calib_labelled.sort("window_index")["is_nominal"]).to_numpy()
        scores = model.score(calib_values)
        n = min(len(scores), len(calib_window_y))
        return scores[:n], calib_window_y[:n]

    def _calib_window_threshold(self, model, calib_df):
        scores, calib_window_y = self._calib_window_scores(model, calib_df)
        return _threshold(scores, calib_window_y)

    def refit(self, fit_values, calib_df, eval_values):
        model = WindowedIsolationForest(self.spec, n_estimators=100, contamination=0.05).fit(fit_values)
        thr = self._calib_window_threshold(model, calib_df)
        window_pred = model.score(eval_values) > thr
        return broadcast_window_pred_to_rows(window_pred, eval_values.shape[0], self.spec)

    def online(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = WindowedIsolationForest(self.spec, n_estimators=100, contamination=0.05).fit(fit_values)
        else:
            self.online_model.partial_fit(fit_values, n_new_estimators=100)
        thr = self._calib_window_threshold(self.online_model, calib_df)
        window_pred = self.online_model.score(eval_values) > thr
        return broadcast_window_pred_to_rows(window_pred, eval_values.shape[0], self.spec)

    def online_raw(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = WindowedIsolationForest(self.spec, n_estimators=100, contamination=0.05).fit(fit_values)
        else:
            self.online_model.partial_fit(fit_values, n_new_estimators=100)
        calib_scores, calib_window_y = self._calib_window_scores(self.online_model, calib_df)
        return calib_scores, calib_window_y, self.online_model.score(eval_values), True


class AutoencoderRunner(ModelRunner):
    name = "Autoencoder"

    def __init__(self, target_channels, labels_long, config, n_channels):
        super().__init__(target_channels, labels_long, config)
        self.n_channels = n_channels

    def refit(self, fit_values, calib_df, eval_values):
        model = Autoencoder(n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8)
        model.fit(fit_values, epochs=20, batch_size=1024, verbose=False)
        calib_values, calib_y = self._calib_arrays(calib_df)
        thr = _threshold(model.score(calib_values), calib_y)
        return model.predict(eval_values, threshold=thr)

    def online(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = Autoencoder(n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8)
        # Fewer epochs per fold than refit's fresh 20 -- this model accumulates training
        # across ~20 folds (an effective ~200 epoch-equivalents on cumulative data),
        # while refit only ever sees 20 epochs on a single month before being discarded.
        self.online_model.partial_fit(fit_values, epochs=10, batch_size=1024, verbose=False)
        calib_values, calib_y = self._calib_arrays(calib_df)
        thr = _threshold(self.online_model.score(calib_values), calib_y)
        return self.online_model.predict(eval_values, threshold=thr)

    def online_raw(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = Autoencoder(n_channels=self.n_channels, hidden_sizes=(32, 16), latent_size=8)
        self.online_model.partial_fit(fit_values, epochs=10, batch_size=1024, verbose=False)
        calib_values, calib_y = self._calib_arrays(calib_df)
        return self.online_model.score(calib_values), calib_y, self.online_model.score(eval_values), False


class LSTMRunner(ModelRunner):
    name = "LSTMForecaster"

    def __init__(self, target_channels, labels_long, config, n_channels):
        super().__init__(target_channels, labels_long, config)
        self.n_channels = n_channels

    def _new_model(self):
        # Smaller than LSTMForecaster's own defaults (input_length=100, hidden_size=64)
        # -- kept modest so ~20 folds x 2 variants of real training stays tractable on a
        # laptop CPU; stride=20 (vs. the default 1) similarly bounds the number of
        # training windows built from each ~1-month fold, since stride=1 over a full
        # month would materialize on the order of 100K overlapping windows of shape
        # (input_length, n_channels) at once.
        return LSTMForecaster(n_channels=self.n_channels, input_length=50, hidden_size=32, num_layers=1)

    def refit(self, fit_values, calib_df, eval_values):
        model = self._new_model()
        model.partial_fit(fit_values, epochs=10, batch_size=256, stride=20, verbose=False)
        calib_values = channel_matrix(calib_df, self.target_channels)
        model.calibrate_threshold(calib_values, percentile=99.0)
        return model.predict(eval_values)

    def online(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values, epochs=5, batch_size=256, stride=20, verbose=False)
        calib_values = channel_matrix(calib_df, self.target_channels)
        self.online_model.calibrate_threshold(calib_values, percentile=99.0)
        return self.online_model.predict(eval_values)

    def online_raw(self, fit_values, calib_df, eval_values):
        if self.online_model is None:
            self.online_model = self._new_model()
        self.online_model.partial_fit(fit_values, epochs=5, batch_size=256, stride=20, verbose=False)
        calib_values = channel_matrix(calib_df, self.target_channels)
        # LSTMForecaster's native calibration is percentile-only (no find_best_threshold
        # tuning against labels) -- calib_y=None signals that to the caller, matching
        # online()'s own calibrate_threshold(..., percentile=99.0) call above exactly.
        return self.online_model.score(calib_values), None, self.online_model.score(eval_values), False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--max-folds", type=int, default=None, help="default: no cap, covers the full test period")
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS))
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    requested = args.models.split(",")

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)
    channel_meta = load_channel_metadata(Path(config.raw_data_dir) / "channels.csv")
    target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()
    n_channels = len(target_channels)
    print(f"Mission{args.mission}: {n_channels} target channels")

    chunks_glob = str(Path(config.processed_dir) / "chunks" / "*.parquet")
    lf = pl.scan_parquet(chunks_glob)
    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")

    spec = WindowSpec(length=10, stride=10)
    subsystem_col = "subsystem" if "subsystem" in channel_meta.columns else None
    channel_groups = {}
    if subsystem_col:
        channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta[subsystem_col]))
        for i, ch in enumerate(target_channels):
            channel_groups.setdefault(channel_to_subsystem.get(ch, "unknown"), []).append(i)

    runners: dict[str, ModelRunner] = {}
    if "GlobalSTD" in requested:
        runners["GlobalSTD"] = GlobalSTDRunner(target_channels, labels_long, config)
    if "PCAReconstructor" in requested:
        runners["PCAReconstructor"] = PCARunner(target_channels, labels_long, config)
    if "WindowedIsolationForest" in requested:
        runners["WindowedIsolationForest"] = WindowedIForestRunner(target_channels, labels_long, config, spec)
    if "SubsystemPCAEnsemble" in requested and channel_groups:
        runners["SubsystemPCAEnsemble"] = SubsystemRunner(target_channels, labels_long, config, channel_groups)
    if "Autoencoder" in requested:
        runners["Autoencoder"] = AutoencoderRunner(target_channels, labels_long, config, n_channels)
    if "LSTMForecaster" in requested:
        runners["LSTMForecaster"] = LSTMRunner(target_channels, labels_long, config, n_channels)
    print(f"Models: {list(runners)}")

    # "sliding, 1 month" fold definition: train_start = train_end - 1 month, for every
    # fold -- this IS the "refit from scratch on just the previous month" period the
    # "refit" variant needs, and is also the right-sized new-data slice for "online"'s
    # partial_fit each fold (both variants see the SAME fold boundaries; only what they
    # do with that data differs).
    folds = list(
        generate_walk_forward_folds(config.split.test_start, config.split.test_end, step_months=1, window="sliding", sliding_window_months=1)
    )
    if args.max_folds:
        folds = folds[: args.max_folds]
    print(f"{len(folds)} monthly folds over the FULL test period ({config.split.test_start.date()} .. {config.split.test_end.date()})")

    results: dict[str, dict[str, list[dict]]] = {name: {"refit": [], "online": [], "hybrid": []} for name in runners}
    t_start = time.time()

    for fold in folds:
        train_df = _load_slice(lf, fold.train_start, fold.train_end, target_channels)
        eval_df = _load_slice(lf, fold.eval_start, fold.eval_end, target_channels)
        if train_df.height < 300 or eval_df.height < 200:
            continue

        split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
        fit_df = train_df.filter(pl.col("time") < split_point)
        calib_df = train_df.filter(pl.col("time") >= split_point)
        fit_nominal = filter_nominal_rows(fit_df, labels_long, target_channels)
        if fit_nominal.height < 300 or calib_df.height < 200:
            continue
        fit_values = channel_matrix(fit_nominal, target_channels)
        eval_values = channel_matrix(eval_df, target_channels)
        eval_y = row_is_non_nominal(eval_df, labels_long, target_channels)

        fold_t0 = time.time()
        for name, runner in runners.items():
            t0 = time.time()
            refit_pred = runner.refit(fit_values, calib_df, eval_values)
            t_refit = time.time() - t0
            t0 = time.time()
            online_pred = runner.online(fit_values, calib_df, eval_values)
            t_online = time.time() - t0
            hybrid_pred = refit_pred & online_pred

            results[name]["refit"].append(_metrics(eval_y, refit_pred))
            results[name]["online"].append(_metrics(eval_y, online_pred))
            results[name]["hybrid"].append(_metrics(eval_y, hybrid_pred))
            print(
                f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) {name}: "
                f"refit_f0.5={results[name]['refit'][-1]['f_beta']:.3f} ({t_refit:.1f}s) "
                f"online_f0.5={results[name]['online'][-1]['f_beta']:.3f} ({t_online:.1f}s) "
                f"hybrid_f0.5={results[name]['hybrid'][-1]['f_beta']:.3f}"
            )
        print(f"  -- fold {fold.fold_index} all models: {time.time() - fold_t0:.1f}s")

    print(f"\nTotal walk-forward time: {time.time() - t_start:.1f}s")
    for name in runners:
        for variant in ("refit", "online", "hybrid"):
            agg = aggregate_fold_results(results[name][variant])
            print(f"\n=== {name}-{variant}: {agg['n_folds']} folds ===")
            print("  micro:", agg["micro"])
            print("  macro F0.5:", agg["macro"].get("f_beta"))
            run_id = log_experiment(
                model_name=f"{name}-{variant}",
                mission=args.mission,
                config={"step_months": 1, "fold_definition": "sliding_1_month"},
                metrics={"micro": agg["micro"], "macro": agg["macro"]},
                dataset_scope={"n_folds": agg["n_folds"], "test_start": str(config.split.test_start), "test_end": str(config.split.test_end)},
                notes=f"Full-mission walk-forward: {variant} methodology for {name}, over the entire Mission{args.mission} test period",
                mirror_to_mlflow=args.mlflow,
            )
            print(f"  logged as [{run_id}]")


if __name__ == "__main__":
    main()
