from __future__ import annotations

import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl

DEFAULT_LOG_PATH = Path("data/experiments/results.jsonl")


def log_experiment(
    model_name: str,
    mission: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    dataset_scope: Optional[dict[str, Any]] = None,
    notes: str = "",
    log_path: str | Path = DEFAULT_LOG_PATH,
    mirror_to_mlflow: bool = False,
) -> str:
    """Append one experiment run as a JSON line to `log_path` -- append-only and
    human-readable (unlike a mutable database file), so runs from different sessions
    merge trivially and nothing gets silently overwritten.

    `config`: hyperparameters and any other run-defining settings (window spec, PCA
    variance, epochs, sliding window length, ...). `dataset_scope`: what data the run
    covered (train/eval date ranges, fold index, row counts) -- kept separate from
    `config` since it describes the *data*, not the *model*. `metrics`: whatever
    `evaluation.py` functions returned (window-level, event-level, event-wise, ...).
    Returns the run's unique id for cross-referencing.

    `mirror_to_mlflow`: also send this run to MLflow (see `mlflow_backend.log_to_mlflow`)
    -- opt-in and a no-op if `mlflow` isn't installed, since this JSONL log is the
    project's source of truth regardless. Off by default so existing callers are
    unaffected.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "mission": mission,
        "config": config,
        "dataset_scope": dataset_scope or {},
        "metrics": metrics,
        "notes": notes,
        "host": platform.node(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    if mirror_to_mlflow:
        from esa_adb.experiments.mlflow_backend import log_to_mlflow

        log_to_mlflow(model_name, mission, config, metrics, dataset_scope, notes)

    return run_id


def load_experiments(log_path: str | Path = DEFAULT_LOG_PATH) -> list[dict]:
    """All logged runs as plain dicts (config/metrics keys vary freely across model
    types, so this deliberately doesn't force them into one rigid tabular schema --
    `summarize_experiments` below extracts the common fields for a flat overview).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_experiments(records: Optional[list[dict]] = None, log_path: str | Path = DEFAULT_LOG_PATH) -> pl.DataFrame:
    """Flat summary table: one row per run, with whichever common metric keys are
    present (window/event-level precision, recall, f_beta -- however each run's
    `metrics` dict happens to be nested, most of this project's `evaluation.py`
    functions return dicts shaped like `{"precision":..., "recall":..., "f_beta":...}`
    or wrap several such dicts under named keys like `"window"`/`"event_wise"`).
    """
    if records is None:
        records = load_experiments(log_path)
    if not records:
        return pl.DataFrame()

    rows = []
    for r in records:
        metrics = r.get("metrics", {})
        flat = _flatten_metrics(metrics)
        rows.append(
            {
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "model_name": r["model_name"],
                "mission": r["mission"],
                "notes": r.get("notes", ""),
                **flat,
            }
        )
    return pl.DataFrame(rows)


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    """One level of dict-nesting flattened into `parent_key` columns (e.g.
    `{"window": {"f_beta": 0.3}}` -> `{"window_f_beta": 0.3}`); deeper structures or
    non-scalar values are dropped from the flat summary (still available in the full
    record via `load_experiments`).
    """
    flat = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=key))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            flat[key] = v
    return flat
