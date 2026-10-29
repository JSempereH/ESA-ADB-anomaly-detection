"""Unified benchmark engine: every model x every protocol.

Two protocols share the same runners (`models/runners.py`), the same threshold
calibration and the same alarm post-processing, so their numbers are directly
comparable:

- walk-forward: (this project's primary methodology): monthly (or weekly) folds over
  the mission's test period; per fold, the training period splits 2/3 fit / 1/3
  calibration, the model scores the next fully held-out period. `refit` / `online` /
  `hybrid` history variants as in the README.
- paper-split: the ESA-ADB paper's own fixed train/val/test split, for numbers
  comparable to the paper's published corrected event-wise F0.5.

The headline metric everywhere is the paper's **corrected event-wise F0.5**
(`evaluation.evaluate_event_wise_corrected`), reported per fold (macro) and pooled over
the full evaluated timeline (predictions from all folds concatenated and segmented
once, so an alarm crossing a fold boundary counts as one detection). Thresholds are
calibrated by maximizing that same metric on calibration data
(`find_best_threshold_event_wise`), with `calibration="row"` available as the legacy
row-level F0.5 objective for A/B comparison.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from esa_adb.config import MissionConfig, load_mission_config
from esa_adb.evaluation import (
    DEFAULT_THRESHOLD_QUANTILES,
    anomalies_only,
    evaluate_event_wise,
    evaluate_event_wise_corrected,
    find_best_threshold,
    find_best_threshold_event_wise,
)
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import channel_matrix
from esa_adb.models.postprocess import PostprocessSpec, ewma
from esa_adb.models.runners import ScoreRunner, build_runners
from esa_adb.models.walk_forward import generate_walk_forward_folds
from esa_adb.preprocessing.labeling import filter_nominal_rows, row_is_non_nominal
from esa_adb.preprocessing.windowing import WindowSpec

DEFAULT_MODELS = ["GlobalSTD", "PCAReconstructor", "SubsystemPCAEnsemble", "WindowedIsolationForest", "HalfSpaceTrees", "Autoencoder", "LSTMForecaster"]


@dataclass
class BenchmarkData:
    config: MissionConfig
    mission: int
    lf: pl.LazyFrame
    labels_long: pl.DataFrame
    events: pl.DataFrame
    target_channels: list[str]
    channel_groups: dict
    spec: WindowSpec = field(default_factory=lambda: WindowSpec(length=10, stride=10))
    # Model INPUT columns: the labeled target channels plus any auxiliary features
    # (e.g. telecommand activity). Labels/events always refer to `target_channels` only.
    input_channels: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.input_channels:
            self.input_channels = list(self.target_channels)


TELECOMMAND_FEATURES = ["tc_count", "tc_recency"]


def load_benchmark_data(
    mission: int,
    config_dir: str = "configs",
    exclude_channels: Optional[list[str]] = None,
    with_telecommands: bool = False,
) -> BenchmarkData:
    config = load_mission_config(Path(config_dir) / f"mission{mission}.yaml")
    channel_meta = load_channel_metadata(Path(config.raw_data_dir) / "channels.csv")
    target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()
    if exclude_channels:
        target_channels = [c for c in target_channels if c not in exclude_channels]

    lf = pl.scan_parquet(str(Path(config.processed_dir) / "chunks" / "*.parquet"))
    input_channels = list(target_channels)
    if with_telecommands:
        tc_path = Path(config.processed_dir) / "telecommands.parquet"
        if not tc_path.exists():
            raise FileNotFoundError(f"{tc_path} not found -- run scripts/build_telecommand_features.py --mission {mission} first")
        lf = lf.join(pl.scan_parquet(str(tc_path)).select(["time", *TELECOMMAND_FEATURES]), on="time", how="left")
        input_channels += TELECOMMAND_FEATURES
    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
    events = pl.read_csv(Path(config.interim_dir) / f"anomalies_mission{mission}.csv", try_parse_dates=True)
    events = events.filter(pl.col("Channel").is_in(target_channels))

    subsystem_col = "subsystem" if "subsystem" in channel_meta.columns else None
    channel_groups: dict = {}
    if subsystem_col:
        channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta[subsystem_col]))
        for i, ch in enumerate(target_channels):
            channel_groups.setdefault(channel_to_subsystem.get(ch, "unknown"), []).append(i)

    return BenchmarkData(
        config=config, mission=mission, lf=lf, labels_long=labels_long, events=events,
        target_channels=target_channels, channel_groups=channel_groups, input_channels=input_channels,
    )


def _load_slice(data: BenchmarkData, start, end) -> pl.DataFrame:
    return (
        data.lf.filter((pl.col("time") >= start) & (pl.col("time") < end))
        .select(["time", *data.input_channels])
        .sort("time")
        .collect()
    )


def _row_metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta**2
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f_beta": f_beta, "beta": beta}


def _calibrate(
    calib_scores: np.ndarray,
    calib_times: np.ndarray,
    calib_y: Optional[np.ndarray],
    events: pl.DataFrame,
    pp: PostprocessSpec,
    calibration: str,
    alarm_budget: float = 20.0,
    max_alarm_time_fraction: float = 0.02,
) -> tuple[float, Optional[float]]:
    """Returns (threshold, smoothing alpha). `calibration="event"` sweeps quantile
    candidates x smoothing alphas against corrected event-wise F0.5 with the alarm
    post-processing in the loop; `"row"` is the legacy row-level F0.5 objective
    (no smoothing -- the historical behavior being A/B'd against).

    `calibration="budget"` is the 100% UNSUPERVISED, production-deployable mode: no
    labels, no annotated events -- the operator states TWO label-free constraints ("I
    can review ~`alarm_budget` alarms per 30 days" and "at most
    `max_alarm_time_fraction` of the timeline may sit inside an alarm") and the
    threshold is the LOWEST quantile candidate satisfying both on the calibration
    window (lowest = maximum sensitivity the constraints allow). The time-fraction cap
    exists because a handful of very LONG alarms can respect the count budget while
    swallowing the whole quiet timeline. Smoothing is fixed to `pp.smooth_alphas[0]`
    since, without labels, there is no criterion to pick between alphas. Labels are
    then only ever used to *measure* the result, never to operate.
    """
    if calibration == "budget":
        from esa_adb.models.postprocess import _segments

        alpha = pp.smooth_alphas[0]
        smoothed = ewma(calib_scores, alpha)
        span_days = max((calib_times.max() - calib_times.min()) / np.timedelta64(1, "D"), 1e-9)
        max_alarms = alarm_budget * span_days / 30.0
        candidates = np.unique(np.quantile(smoothed, DEFAULT_THRESHOLD_QUANTILES))
        best = float(candidates[-1])  # quietest fallback if even it violates the budget
        for thr in candidates[::-1]:  # quietest first: keep descending while within budget
            pred = pp.apply(smoothed > thr)
            if len(_segments(pred)[0]) > max_alarms or pred.mean() > max_alarm_time_fraction:
                break
            best = float(thr)
        return best, alpha

    if calibration == "row":
        if calib_y is not None and calib_y.sum() > 0:
            tuned = find_best_threshold(calib_scores, calib_y, beta=0.5)
            if tuned["f_beta"] > 0:
                return tuned["threshold"], None
        return float(np.percentile(calib_scores, 99.0)), None

    best_thr, best_alpha, best_f = float(np.percentile(calib_scores, 99.9)), None, -1.0
    for alpha in pp.smooth_alphas:
        smoothed = ewma(calib_scores, alpha)
        res = find_best_threshold_event_wise(smoothed, calib_times, events, beta=0.5, postprocess=pp.apply)
        if res["f_beta"] > best_f:
            best_thr, best_alpha, best_f = res["threshold"], alpha, res["f_beta"]
    return best_thr, best_alpha


def _predict(eval_scores: np.ndarray, threshold: float, alpha: Optional[float], pp: PostprocessSpec) -> np.ndarray:
    return pp.apply(ewma(eval_scores, alpha) > threshold)


def _pooled_metrics(fold_records: list[dict], events: pl.DataFrame) -> dict:
    """Pooled (concatenate every fold's predictions, segment once over the full
    timeline) + macro (mean of per-fold corrected event-wise F0.5) aggregation.

    The macro mean only counts folds with `n_events > 0`. A fold with zero annotated
    events AND zero detections hits `evaluate_event_wise`'s explicit 0/0 fallback (0.0 --
    neither a real precision nor recall exists to average), which is fine for the
    *pooled* number (that fold just contributes no TP/FP/FN to the single full-timeline
    score) but would silently drag the *macro* mean down if included: a model correctly
    staying silent through a quiet month would score identically to one that missed a
    real event that month. Excluding those folds instead of zeroing them keeps macro
    measuring "how well did it do when there was something to detect" -- the same folds
    `analyze_paper_stats.py`'s Wilcoxon tests should exclude for the same reason.
    """
    pred = np.concatenate([r["pred"] for r in fold_records])
    times = np.concatenate([r["times"] for r in fold_records])
    y = np.concatenate([r["y"] for r in fold_records])
    ew = evaluate_event_wise(pred, times, events, beta=0.5)
    ewc = evaluate_event_wise_corrected(pred, times, events, beta=0.5)
    scored = [r["event_wise_corrected"]["f_beta"] for r in fold_records if r["event_wise_corrected"]["n_events"] > 0]

    # Same pooled/macro/per-fold treatment, but against Category=="anomaly" events only
    # (see evaluation.anomalies_only) -- the paper's own "results for anomalies alone"
    # (Supplementary 4.2), answering "are we catching the real anomalies specifically"
    # rather than the rare-nominal-event-dominated default metric above.
    anomaly_events = anomalies_only(events)
    ewc_anomalies = evaluate_event_wise_corrected(pred, times, anomaly_events, beta=0.5)
    fold_ewc_anomalies = [evaluate_event_wise_corrected(r["pred"], r["times"], anomaly_events, beta=0.5) for r in fold_records]
    scored_anomalies = [m["f_beta"] for m in fold_ewc_anomalies if m["n_events"] > 0]

    return {
        "pooled_event_wise": ew,
        "pooled_event_wise_corrected": ewc,
        "macro_event_wise_corrected_f0.5": float(np.mean(scored)) if scored else 0.0,
        "macro_n_folds_scored": len(scored),
        "pooled_event_wise_corrected_anomalies_only": ewc_anomalies,
        "macro_f0.5_anomalies_only": float(np.mean(scored_anomalies)) if scored_anomalies else 0.0,
        "macro_n_folds_scored_anomalies_only": len(scored_anomalies),
        "row_micro": _row_metrics(y, pred),
        "n_folds": len(fold_records),
        # Per-fold corrected event-wise summaries: the paired samples that fold-level
        # significance tests (Wilcoxon online-vs-refit etc.) and cold-start analyses
        # need -- summary statistics alone can't support either. eval_start/eval_end
        # (derived from this fold's own eval-window timestamps, so it works for every
        # caller regardless of protocol) are what let a later time-series plot line up
        # each fold's metrics against the calendar -- essential once a run can span the
        # full mission (`full_history=True`) rather than a short, roughly-uniform test
        # window.
        "folds": [
            {
                "fold_index": i,
                "eval_start": str(r["times"].min()),
                "eval_end": str(r["times"].max()),
                "f_beta": r["event_wise_corrected"]["f_beta"],
                "precision": r["event_wise_corrected"]["precision"],
                "recall": r["event_wise_corrected"]["recall"],
                "n_detections": r["event_wise_corrected"]["n_detections"],
                "n_events": r["event_wise_corrected"]["n_events"],
                "n_events_detected": r["event_wise_corrected"]["n_events_detected"],
                "f_beta_anomalies_only": fold_ewc_anomalies[i]["f_beta"],
                "recall_anomalies_only": fold_ewc_anomalies[i]["recall"],
                "n_events_anomalies_only": fold_ewc_anomalies[i]["n_events"],
                "n_events_detected_anomalies_only": fold_ewc_anomalies[i]["n_events_detected"],
            }
            for i, r in enumerate(fold_records)
        ],
    }



def run_channel_aware(
    data: BenchmarkData,
    model_names: list[str],
    step_months: int = 1,
    calibration: str = "event",
    pp: Optional[PostprocessSpec] = None,
    alarm_budget: float = 20.0,
    max_alarm_time_fraction: float = 0.02,
    calib_buffer_folds: int = 6,
    max_folds: Optional[int] = None,
    mlflow: bool = False,
    tag: str = "",
    verbose: bool = True,
) -> dict:
    """Channel-attributed walk-forward evaluation (online history only).

    The global benchmark raises ONE mission-level alarm stream and credits it against
    every channel's events -- the paper instead scores *per-channel* predictions and
    aggregates channel -> subsystem -> mission. This engine closes that gap for models
    exposing per-channel scores: each of the 47 target channels gets its OWN threshold
    (calibrated per fold against only that channel's events, or its own alarm budget
    when `calibration="budget"`), its own post-processed alarm stream, and its own
    pooled corrected event-wise F0.5 against only its own annotated events.

    Reported aggregates:
    - `channel_aware_macro_f0.5`: mean over channels with >=1 annotated event in the
      test period (channels with no events have undefined recall and are excluded,
      count reported).
    - `subsystem_aware_macro_f0.5`: mean within each subsystem first, then across
      subsystems -- the paper's hierarchical flavor.
    - `global_or`: the OR across channels' alarms scored the mission-level way, so the
      cost/benefit of attribution vs. one global alarm stream is measurable directly.

    Per-channel thresholds calibrate on a rolling buffer of the last
    `calib_buffer_folds` folds' calibration scores, not just the current fold's ~10
    days: a single channel rarely has any annotated event in 10 days, which otherwise
    forces the quiet fallback threshold on most channels and silences them (confirmed
    empirically: 0.279 channel-macro with per-fold calibration vs. buffered). Buffered
    scores come from the model state at the time each fold was calibrated -- an
    approximation that avoids re-scoring months of history every fold, acceptable
    because the online models drift slowly.
    """
    pp = pp or PostprocessSpec()
    runners = build_runners(model_names, len(data.input_channels), data.channel_groups, data.spec)
    folds = list(
        generate_walk_forward_folds(
            data.config.split.test_start, data.config.split.test_end,
            step_months=step_months, window="sliding", sliding_window_months=step_months,
        )
    )
    if max_folds:
        folds = folds[:max_folds]
    n_ch = len(data.target_channels)
    events_by_channel = {ch: data.events.filter(pl.col("Channel") == ch) for ch in data.target_channels}
    if verbose:
        print(f"channel-aware walk-forward: {len(folds)} folds, {n_ch} channels, models={list(runners)}, calibration={calibration}")

    summary: dict[str, dict] = {}
    for name, runner in runners.items():
        t_start = time.time()
        ch_preds: list[list[np.ndarray]] = [[] for _ in range(n_ch)]
        or_records: list[dict] = []
        times_all: list[np.ndarray] = []
        calib_buffer: list[tuple[np.ndarray, np.ndarray]] = []  # (calib_pc, calib_times) of recent folds

        for fold in folds:
            train_df = _load_slice(data, fold.train_start, fold.train_end)
            eval_df = _load_slice(data, fold.eval_start, fold.eval_end)
            if train_df.height < 300 or eval_df.height < 200:
                continue
            split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
            fit_df = train_df.filter(pl.col("time") < split_point)
            calib_df = train_df.filter(pl.col("time") >= split_point)
            fit_nominal = filter_nominal_rows(fit_df, data.labels_long, data.target_channels)
            if fit_nominal.height < 300 or calib_df.height < 200:
                continue

            runner.update(channel_matrix(fit_nominal, data.input_channels))
            calib_pc = runner.score_rows_per_channel(channel_matrix(calib_df, data.input_channels))[:, :n_ch]
            eval_pc = runner.score_rows_per_channel(channel_matrix(eval_df, data.input_channels))[:, :n_ch]
            calib_times = calib_df["time"].to_numpy()
            eval_times = eval_df["time"].to_numpy()
            times_all.append(eval_times)

            calib_buffer.append((calib_pc, calib_times))
            if len(calib_buffer) > calib_buffer_folds:
                calib_buffer.pop(0)
            buffer_pc = np.concatenate([b[0] for b in calib_buffer])
            buffer_times = np.concatenate([b[1] for b in calib_buffer])

            or_pred = np.zeros(eval_pc.shape[0], dtype=bool)
            for c, ch in enumerate(data.target_channels):
                thr, alpha = _calibrate(
                    buffer_pc[:, c], buffer_times, None, events_by_channel[ch], pp, calibration,
                    alarm_budget, max_alarm_time_fraction,
                )
                pred_c = _predict(eval_pc[:, c], thr, alpha, pp)
                ch_preds[c].append(pred_c)
                or_pred |= pred_c
            eval_y = row_is_non_nominal(eval_df, data.labels_long, data.target_channels)
            or_ewc = evaluate_event_wise_corrected(or_pred, eval_times, data.events, beta=0.5)
            or_records.append({"pred": or_pred, "times": eval_times, "y": eval_y, "event_wise_corrected": or_ewc})
            if verbose:
                print(f"fold {fold.fold_index} {name}: OR-of-channels={or_ewc['f_beta']:.3f}({or_ewc['n_detections']}det)")

        times_cat = np.concatenate(times_all)
        per_channel: dict[str, dict] = {}
        for c, ch in enumerate(data.target_channels):
            pred_cat = np.concatenate(ch_preds[c])
            ewc = evaluate_event_wise_corrected(pred_cat, times_cat, events_by_channel[ch], beta=0.5)
            per_channel[ch] = {"f_beta": ewc["f_beta"], "n_events": ewc["n_events"], "n_detections": ewc["n_detections"], "recall": ewc["recall"], "precision": ewc["precision"]}

        scored = {ch: m for ch, m in per_channel.items() if m["n_events"] > 0}
        channel_macro = float(np.mean([m["f_beta"] for m in scored.values()])) if scored else 0.0
        subsystem_scores = {}
        index_to_channel = dict(enumerate(data.target_channels))
        for subsystem, indices in data.channel_groups.items():
            fs = [per_channel[index_to_channel[i]]["f_beta"] for i in indices if index_to_channel[i] in scored]
            if fs:
                subsystem_scores[str(subsystem)] = float(np.mean(fs))
        subsystem_macro = float(np.mean(list(subsystem_scores.values()))) if subsystem_scores else 0.0

        agg = {
            "channel_aware_macro_f0.5": channel_macro,
            "subsystem_aware_macro_f0.5": subsystem_macro,
            "subsystem_scores": subsystem_scores,
            "n_channels_scored": len(scored),
            "n_channels_total": n_ch,
            "per_channel": per_channel,
            "global_or": _pooled_metrics(or_records, data.events),
        }
        summary[name] = agg
        run_id = log_experiment(
            model_name=f"{name}-channel-aware",
            mission=data.mission,
            config={
                "protocol": "walk-forward-channel-aware", "step": f"{step_months}mo", "calibration": calibration,
                "alarm_budget": alarm_budget if calibration == "budget" else None,
                "merge_gap_rows": pp.merge_gap_rows, "min_duration_rows": pp.min_duration_rows,
                "smooth_alphas": [a for a in pp.smooth_alphas], "n_channels": n_ch,
                "supervision": _supervision_descriptor(calibration, False),
            },
            metrics={k: v for k, v in agg.items() if k != "per_channel"} | {"per_channel": per_channel},
            dataset_scope={"n_folds": len(or_records), "test_start": str(data.config.split.test_start), "test_end": str(data.config.split.test_end)},
            notes=tag or f"Channel-attributed walk-forward for {name}: one threshold/alarm stream per channel",
            mirror_to_mlflow=mlflow,
        )
        if verbose:
            g = agg["global_or"]["pooled_event_wise_corrected"]
            print(
                f"=== {name}-channel-aware: channel-macro F0.5={channel_macro:.3f} "
                f"subsystem-macro={subsystem_macro:.3f} ({len(scored)}/{n_ch} channels scored) "
                f"OR-of-channels global F0.5={g['f_beta']:.3f} [{time.time() - t_start:.1f}s, {run_id}]"
            )
    return summary


def _supervision_descriptor(calibration: str, fit_on_all_rows: bool) -> str:
    """One explicit sentence per run stating exactly where labels enter the pipeline --
    logged into every experiment's config so runs with different supervision assumptions
    are never compared as if they were equivalent."""
    fit_part = "fit=all-rows (no label filter)" if fit_on_all_rows else "fit=label-filtered nominal rows"
    cal_part = {
        "event": "threshold=label-tuned (corrected event-wise F0.5 on annotated events)",
        "row": "threshold=label-tuned (row-level F0.5 on annotated rows)",
        "budget": "threshold=unsupervised (alarm budget + time-fraction cap, no labels)",
    }[calibration]
    fully = calibration == "budget" and fit_on_all_rows
    return f"{fit_part}; {cal_part}" + ("; labels used ONLY for final evaluation" if fully else "")

def run_walk_forward(
    data: BenchmarkData,
    model_names: list[str],
    variants: tuple[str, ...] = ("refit", "online", "hybrid"),
    step_months: int = 1,
    step_weeks: Optional[int] = None,
    calibration: str = "event",
    pp: Optional[PostprocessSpec] = None,
    max_folds: Optional[int] = None,
    mlflow: bool = False,
    tag: str = "",
    ae_replay_rows: int = 0,
    ae_online_epochs: int = 10,
    ae_online_lr: float = 1e-3,
    alarm_budget: float = 20.0,
    max_alarm_time_fraction: float = 0.02,
    fit_on_all_rows: bool = False,
    warmup_folds: int = 0,
    ae_seed: int = 42,
    full_history: bool = False,
    return_records: bool = False,
    verbose: bool = True,
) -> dict | tuple[dict, dict]:
    """`full_history=False` (default, every existing logged run): folds only cover
    `config.split.test_start -> test_end` -- the mission's first half (`train_start ->
    test_start`, the paper's own train+val period) is never walked forward, so the
    `online` variant's persisted model is "born" already mid-mission with zero prior
    exposure. `full_history=True` starts folds at `config.split.train_start` instead --
    the ENTIRE mission becomes one continuous walk-forward from the satellite's very
    first recorded month, with no period held out as a separate paper-comparable split.
    Fold width is unaffected either way (still `sliding_window_months`/`sliding_window_weeks`
    trailing data per fold); this only moves where fold 0 begins, which roughly doubles
    `n_folds` for both missions (train+test happen to be ~50/50 splits) and thus runtime.
    """
    pp = pp or PostprocessSpec()
    runners = build_runners(
        model_names, len(data.input_channels), data.channel_groups, data.spec,
        ae_replay_rows=ae_replay_rows, ae_online_epochs=ae_online_epochs, ae_online_lr=ae_online_lr, ae_seed=ae_seed,
    )
    need_refit = "refit" in variants or "hybrid" in variants
    need_online = "online" in variants or "hybrid" in variants

    fold_start = data.config.split.train_start if full_history else data.config.split.test_start
    folds = list(
        generate_walk_forward_folds(
            fold_start, data.config.split.test_end,
            step_months=step_months, step_weeks=step_weeks,
            window="sliding", sliding_window_months=step_months, sliding_window_weeks=step_weeks,
        )
    )
    if max_folds:
        folds = folds[:max_folds]
    step_desc = f"{step_weeks}w" if step_weeks else f"{step_months}mo"
    if verbose:
        print(f"walk-forward: {len(folds)} folds of {step_desc}, models={list(runners)}, calibration={calibration}")

    records: dict[str, dict[str, list[dict]]] = {name: {v: [] for v in variants} for name in runners}
    t_start = time.time()

    for fold in folds:
        train_df = _load_slice(data, fold.train_start, fold.train_end)
        eval_df = _load_slice(data, fold.eval_start, fold.eval_end)
        if train_df.height < 300 or eval_df.height < 200:
            continue
        split_point = train_df["time"].min() + (train_df["time"].max() - train_df["time"].min()) * 2 / 3
        fit_df = train_df.filter(pl.col("time") < split_point)
        calib_df = train_df.filter(pl.col("time") >= split_point)
        # fit_on_all_rows: skip the label-informed nominal filter -- the truly
        # unsupervised production condition, where anomalous rows pollute the fit.
        fit_nominal = fit_df if fit_on_all_rows else filter_nominal_rows(fit_df, data.labels_long, data.target_channels)
        if fit_nominal.height < 300 or calib_df.height < 200:
            continue

        fit_values = channel_matrix(fit_nominal, data.input_channels)
        calib_values = channel_matrix(calib_df, data.input_channels)
        eval_values = channel_matrix(eval_df, data.input_channels)
        calib_times = calib_df["time"].to_numpy()
        eval_times = eval_df["time"].to_numpy()
        calib_y = row_is_non_nominal(calib_df, data.labels_long, data.target_channels)
        eval_y = row_is_non_nominal(eval_df, data.labels_long, data.target_channels)

        for name, runner in runners.items():
            t0 = time.time()
            preds: dict[str, np.ndarray] = {}
            if need_refit:
                calib_s, eval_s = runner.refit_scores(fit_values, calib_values, eval_values)
                thr, alpha = _calibrate(calib_s, calib_times, calib_y, data.events, pp, calibration, alarm_budget, max_alarm_time_fraction)
                preds["refit"] = _predict(eval_s, thr, alpha, pp)
            if need_online:
                calib_s, eval_s = runner.online_scores(fit_values, calib_values, eval_values)
                thr, alpha = _calibrate(calib_s, calib_times, calib_y, data.events, pp, calibration, alarm_budget, max_alarm_time_fraction)
                preds["online"] = _predict(eval_s, thr, alpha, pp)
            if "hybrid" in variants:
                # AND of the two variants' post-processed alarms, re-fused/re-filtered:
                # the intersection can leave slivers shorter than the persistence rule.
                preds["hybrid"] = pp.apply(preds["refit"] & preds["online"])

            # Warmup folds (cold-start suppression): the online model trains and the
            # threshold calibrates exactly as always, but the fold's alarms are
            # discarded rather than scored -- a deployed system's "commissioning
            # period" during which operators don't act on a still-miscalibrated model.
            if fold.fold_index < warmup_folds:
                if verbose:
                    print(f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) {name}: warmup (not scored)")
                continue

            line = []
            for variant in variants:
                ewc = evaluate_event_wise_corrected(preds[variant], eval_times, data.events, beta=0.5)
                records[name][variant].append(
                    {"pred": preds[variant], "times": eval_times, "y": eval_y, "event_wise_corrected": ewc}
                )
                line.append(f"{variant}={ewc['f_beta']:.3f}({ewc['n_detections']}det)")
            if verbose:
                print(f"fold {fold.fold_index} ({fold.eval_start.date()}..{fold.eval_end.date()}) {name}: " + " ".join(line) + f" [{time.time() - t0:.1f}s]")

    summary: dict[str, dict[str, dict]] = {}
    for name in records:
        summary[name] = {}
        for variant in variants:
            if not records[name][variant]:
                continue
            agg = _pooled_metrics(records[name][variant], data.events)
            summary[name][variant] = agg
            run_id = log_experiment(
                model_name=f"{name}-{variant}",
                mission=data.mission,
                config={
                    "protocol": "walk-forward", "step": step_desc, "calibration": calibration, "alarm_budget": alarm_budget if calibration == "budget" else None, "fit_on_all_rows": fit_on_all_rows, "warmup_folds": warmup_folds, "ae_seed": ae_seed,
                    "supervision": _supervision_descriptor(calibration, fit_on_all_rows),
                    "merge_gap_rows": pp.merge_gap_rows, "min_duration_rows": pp.min_duration_rows,
                    "smooth_alphas": [a for a in pp.smooth_alphas], "n_channels": len(data.target_channels),
                    "full_history": full_history,
                } | runners[name].run_config(),
                metrics=agg,
                dataset_scope={"n_folds": agg["n_folds"], "test_start": str(fold_start), "test_end": str(data.config.split.test_end)},
                notes=tag or f"run_benchmark walk-forward ({step_desc} folds, {calibration}-calibrated thresholds, alarm postprocessing)",
                mirror_to_mlflow=mlflow,
            )
            if verbose:
                ewc = agg["pooled_event_wise_corrected"]
                ewc_anom = agg["pooled_event_wise_corrected_anomalies_only"]
                print(
                    f"=== {name}-{variant}: pooled corrected event-wise F0.5={ewc['f_beta']:.3f} "
                    f"(precision={ewc['precision']:.3f}, recall={ewc['recall']:.3f}, "
                    f"detections={ewc['n_detections']}, events {ewc['n_events_detected']}/{ewc['n_events']}) "
                    f"macro={agg['macro_event_wise_corrected_f0.5']:.3f} | "
                    f"anomalies-only: F0.5={ewc_anom['f_beta']:.3f}, recall={ewc_anom['recall']:.3f}, "
                    f"anomalies {ewc_anom['n_events_detected']}/{ewc_anom['n_events']} [{run_id}]"
                )
    if verbose:
        print(f"total: {time.time() - t_start:.1f}s")
    if return_records:
        return summary, records
    return summary


def run_adaptive_cadence(
    data: BenchmarkData,
    model_name: str = "PCAReconstructor",
    strategies: tuple[str, ...] = ("every-1", "every-4", "every-8", "adaptive", "never"),
    step_weeks: int = 1,
    init_weeks: int = 4,
    calib_weeks: int = 4,
    ks_threshold: float = 0.15,
    pp: Optional[PostprocessSpec] = None,
    mlflow: bool = False,
    tag: str = "",
    verbose: bool = True,
) -> dict:
    """Prequential ("test-then-train") comparison of model-update cadences.

    The walk-forward engine retrains on a fixed calendar schedule; this asks the
    follow-up question the project's own README raises -- *should* the cadence be fixed,
    or driven by the data? Every strategy scores each new week with whatever model it
    currently has (so evaluation is always on unseen data), and only differs in WHEN it
    calls `partial_fit` + recalibrates its threshold afterwards:

    - `every-N`: update every N weeks (N=1 is the classic always-update online model).
    - `adaptive`: update only when the two-sample Kolmogorov-Smirnov statistic between
      this week's score distribution and the scores at the last update exceeds
      `ks_threshold` -- i.e. "the model's own outputs stopped looking like they did when
      the threshold was calibrated", an unsupervised drift signal (METER-style concept:
      adapt the model on drift, not just the threshold).
    - `never`: fit once on the init period, never touched again (ablation floor).

    All strategies share identical init (fit on the first `init_weeks - 1` weeks'
    nominal rows, calibrate on the following week) and identical alarm post-processing.
    Threshold recalibration at update time rescores the last `calib_weeks` weeks with
    the just-updated model (a multi-week calibration buffer -- the README's own open
    question about one month being too noisy a calibration signal).

    Reported per strategy: pooled corrected event-wise F0.5 over every evaluated week,
    plus `n_updates` -- the accuracy-vs-compute trade the cadence question is about.
    """
    from scipy.stats import ks_2samp

    pp = pp or PostprocessSpec()
    rng = np.random.default_rng(42)
    step = relativedelta(weeks=step_weeks)

    fit_end = data.config.split.test_start + relativedelta(weeks=(init_weeks - 1) * step_weeks)
    calib_end = data.config.split.test_start + relativedelta(weeks=init_weeks * step_weeks)
    init_fit_df = _load_slice(data, data.config.split.test_start, fit_end)
    init_calib_df = _load_slice(data, fit_end, calib_end)
    init_fit_nominal = filter_nominal_rows(init_fit_df, data.labels_long, data.target_channels)

    def _subsample(x: np.ndarray, n: int = 5000) -> np.ndarray:
        return x if x.shape[0] <= n else x[rng.choice(x.shape[0], n, replace=False)]

    class _State:
        def __init__(self):
            runners = build_runners([model_name], len(data.input_channels), data.channel_groups, data.spec)
            self.runner = runners[model_name]
            self.runner.update(channel_matrix(init_fit_nominal, data.input_channels))
            calib_values = channel_matrix(init_calib_df, data.input_channels)
            calib_scores = self.runner.score_rows(calib_values)
            calib_y = row_is_non_nominal(init_calib_df, data.labels_long, data.target_channels)
            self.threshold, self.alpha = _calibrate(calib_scores, init_calib_df["time"].to_numpy(), calib_y, data.events, pp, "event")
            self.ref_scores = _subsample(calib_scores)
            self.n_updates = 1
            self.records: list[dict] = []
            self.drift_history: list[float] = []

    states = {s: _State() for s in strategies}
    calib_buffer: list[pl.DataFrame] = [init_calib_df]

    week_idx = 0
    cursor = calib_end
    while cursor + step <= data.config.split.test_end:
        week_df = _load_slice(data, cursor, cursor + step)
        cursor = cursor + step
        week_idx += 1
        if week_df.height < 200:
            continue
        values = channel_matrix(week_df, data.input_channels)
        times = week_df["time"].to_numpy()
        y = row_is_non_nominal(week_df, data.labels_long, data.target_channels)
        week_nominal = filter_nominal_rows(week_df, data.labels_long, data.target_channels)

        calib_buffer.append(week_df)
        if len(calib_buffer) > calib_weeks:
            calib_buffer.pop(0)
        buffer_df = pl.concat(calib_buffer, how="vertical")

        for strategy, st in states.items():
            scores = st.runner.score_rows(values)
            pred = _predict(scores, st.threshold, st.alpha, pp)
            ewc = evaluate_event_wise_corrected(pred, times, data.events, beta=0.5)
            st.records.append({"pred": pred, "times": times, "y": y, "event_wise_corrected": ewc})

            drift = float(ks_2samp(_subsample(scores), st.ref_scores).statistic)
            st.drift_history.append(drift)
            if strategy == "never":
                do_update = False
            elif strategy == "adaptive":
                do_update = drift > ks_threshold
            else:
                every = int(strategy.split("-")[1])
                do_update = week_idx % every == 0

            if do_update and week_nominal.height >= 200:
                st.runner.update(channel_matrix(week_nominal, data.input_channels))
                buffer_values = channel_matrix(buffer_df, data.input_channels)
                buffer_scores = st.runner.score_rows(buffer_values)
                buffer_y = row_is_non_nominal(buffer_df, data.labels_long, data.target_channels)
                st.threshold, st.alpha = _calibrate(buffer_scores, buffer_df["time"].to_numpy(), buffer_y, data.events, pp, "event")
                st.ref_scores = _subsample(buffer_scores)
                st.n_updates += 1
        if verbose:
            drifts = " ".join(f"{s}:d={st.drift_history[-1]:.2f},u={st.n_updates}" for s, st in states.items())
            print(f"week {week_idx} ({times[0]}..): {drifts}")

    summary: dict[str, dict] = {}
    for strategy, st in states.items():
        agg = _pooled_metrics(st.records, data.events)
        agg["n_updates"] = st.n_updates
        agg["mean_drift"] = float(np.mean(st.drift_history)) if st.drift_history else 0.0
        summary[strategy] = agg
        run_id = log_experiment(
            model_name=f"{model_name}-cadence-{strategy}",
            mission=data.mission,
            config={
                "protocol": "adaptive-cadence", "step_weeks": step_weeks, "init_weeks": init_weeks,
                "calib_weeks": calib_weeks, "ks_threshold": ks_threshold, "n_channels": len(data.target_channels),
            },
            metrics=agg,
            dataset_scope={"n_folds": agg["n_folds"], "test_start": str(data.config.split.test_start), "test_end": str(data.config.split.test_end)},
            notes=tag or f"Prequential cadence comparison ({strategy}) for {model_name}",
            mirror_to_mlflow=mlflow,
        )
        if verbose:
            ewc = agg["pooled_event_wise_corrected"]
            print(
                f"=== {model_name}-cadence-{strategy}: pooled corrected event-wise F0.5={ewc['f_beta']:.3f} "
                f"(precision={ewc['precision']:.3f}, recall={ewc['recall']:.3f}, detections={ewc['n_detections']}) "
                f"updates={agg['n_updates']} mean_drift={agg['mean_drift']:.3f} [{run_id}]"
            )
    return summary


def run_paper_split(
    data: BenchmarkData,
    model_names: list[str],
    calibration: str = "event",
    pp: Optional[PostprocessSpec] = None,
    mlflow: bool = False,
    tag: str = "",
    ae_refit_epochs: int = 30,
    alarm_budget: float = 20.0,
    max_alarm_time_fraction: float = 0.02,
    fit_on_all_rows: bool = False,
    verbose: bool = True,
) -> dict:
    """The paper's chronological split, with either nominal-only or contaminated L2 fitting."""
    pp = pp or PostprocessSpec()
    runners = build_runners(model_names, len(data.input_channels), data.channel_groups, data.spec)
    for runner in runners.values():
        if getattr(runner, "refit_epochs", None) is not None:
            runner.refit_epochs = ae_refit_epochs

    # Each Polars DataFrame is dropped as soon as its NumPy array is extracted --
    # for Mission 1's 7-year train+test periods (~7.4M rows each), holding all four
    # DataFrames (train_raw, train_nominal, val_df, test_df) alongside their NumPy
    # duplicates at once peaks around ~16.5 GB before any model even starts fitting
    # (confirmed OOM-killed at that stage on a 16 GB machine) -- releasing each one
    # immediately keeps at most one DataFrame + its array in memory at a time.
    train_raw = data.lf.filter(pl.col("split") == "train").select(["time", *data.target_channels]).collect()
    train_fit = train_raw if fit_on_all_rows else filter_nominal_rows(train_raw, data.labels_long, data.target_channels)
    del train_raw
    fit_values = channel_matrix(train_fit, data.input_channels)
    del train_fit

    val_df = _load_slice(data, data.config.split.val_start, data.config.split.train_end)
    val_values = channel_matrix(val_df, data.input_channels)
    val_times = val_df["time"].to_numpy()
    val_y = row_is_non_nominal(val_df, data.labels_long, data.target_channels)
    del val_df

    test_df = _load_slice(data, data.config.split.test_start, data.config.split.test_end)
    test_values = channel_matrix(test_df, data.input_channels)
    test_times = test_df["time"].to_numpy()
    test_y = row_is_non_nominal(test_df, data.labels_long, data.target_channels)
    del test_df
    if verbose:
        fit_scope = "all contaminated rows" if fit_on_all_rows else "label-filtered nominal rows"
        print(f"paper-split: fit={fit_values.shape[0]} rows ({fit_scope}), val={val_values.shape[0]}, test={test_values.shape[0]}")

    summary: dict[str, dict] = {}
    for name, runner in runners.items():
        t0 = time.time()
        calib_s, eval_s = runner.refit_scores(fit_values, val_values, test_values)
        thr, alpha = _calibrate(calib_s, val_times, val_y, data.events, pp, calibration, alarm_budget, max_alarm_time_fraction)
        pred = _predict(eval_s, thr, alpha, pp)
        agg = {
            "event_wise": evaluate_event_wise(pred, test_times, data.events, beta=0.5),
            "event_wise_corrected": evaluate_event_wise_corrected(pred, test_times, data.events, beta=0.5),
            "event_wise_corrected_anomalies_only": evaluate_event_wise_corrected(
                pred, test_times, anomalies_only(data.events), beta=0.5
            ),
            "row": _row_metrics(test_y, pred),
            "threshold": thr,
            "smooth_alpha": alpha,
        }
        summary[name] = agg
        run_id = log_experiment(
            model_name=name,
            mission=data.mission,
            config={
                "protocol": "paper-split", "calibration": calibration,
                "fit_on_all_rows": fit_on_all_rows,
                "supervision": _supervision_descriptor(calibration, fit_on_all_rows),
                "merge_gap_rows": pp.merge_gap_rows, "min_duration_rows": pp.min_duration_rows,
                "smooth_alphas": [a for a in pp.smooth_alphas], "n_channels": len(data.target_channels),
            } | runner.run_config(),
            metrics=agg,
            dataset_scope={"test_rows": int(test_values.shape[0]), "n_channels": len(data.target_channels)},
            notes=tag or f"run_benchmark paper-split ({calibration}-calibrated thresholds, alarm postprocessing)",
            mirror_to_mlflow=mlflow,
        )
        if verbose:
            ewc = agg["event_wise_corrected"]
            ewc_anom = agg["event_wise_corrected_anomalies_only"]
            print(
                f"=== {name}: corrected event-wise F0.5={ewc['f_beta']:.3f} "
                f"(precision={ewc['precision']:.3f}, recall={ewc['recall']:.3f}, "
                f"detections={ewc['n_detections']}, events {ewc['n_events_detected']}/{ewc['n_events']}) | "
                f"anomalies-only: F0.5={ewc_anom['f_beta']:.3f}, recall={ewc_anom['recall']:.3f}, "
                f"anomalies {ewc_anom['n_events_detected']}/{ewc_anom['n_events']} "
                f"[{time.time() - t0:.1f}s, {run_id}]"
            )
    return summary
