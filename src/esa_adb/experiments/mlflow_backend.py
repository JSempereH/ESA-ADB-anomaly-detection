from __future__ import annotations

from typing import Any, Optional


def log_to_mlflow(
    model_name: str,
    mission: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    dataset_scope: Optional[dict[str, Any]] = None,
    notes: str = "",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "esa-adb-anomaly-detection",
) -> None:
    """Mirror one experiment run into MLflow, alongside (not instead of) `tracking.py`'s
    append-only JSONL log -- MLflow adds a run-comparison UI and metric charts as run
    volume grows with continuous retraining; the JSONL log stays this project's source
    of truth, since it's zero-dependency, diffable in a PR, and every script already
    writes to it. `mlflow` is not a core dependency (`pip install -e ".[tracking]"` to
    enable) -- this is a no-op with a printed notice if it isn't installed, so calling
    code never needs its own try/except.

    `tracking_uri` defaults to a local `sqlite:///mlflow.db` file -- MLflow 3.x put its
    older plain-filesystem store (`file:./mlruns`) into maintenance mode, so sqlite is
    the supported zero-server local backend now. No server to run; point the UI at it
    directly with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
    """
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed -- skipping MLflow logging (pip install -e '.[tracking]' to enable)")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_name}-mission{mission}"):
        mlflow.set_tags({"mission": str(mission), "model_name": model_name, "notes": notes})
        mlflow.log_params(_flatten_for_params(config, prefix="config"))
        if dataset_scope:
            mlflow.log_params(_flatten_for_params(dataset_scope, prefix="data"))
        mlflow.log_metrics(_flatten_for_metrics(metrics))


def _flatten_for_params(d: dict, prefix: str = "") -> dict[str, str]:
    """MLflow params are stored as strings and have no nesting -- flatten one level
    (mirroring `tracking._flatten_metrics`) and str()-cast every leaf value, including
    lists/dicts that survive as a single param rather than being dropped.
    """
    flat: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_for_params(v, prefix=key))
        else:
            flat[key] = str(v)
    return flat


def _flatten_for_metrics(d: dict, prefix: str = "") -> dict[str, float]:
    """MLflow metrics must be numeric -- flatten nested dicts one level at a time and
    keep only int/float/bool leaves (matches what `tracking._flatten_metrics` keeps,
    minus strings, which aren't valid MLflow metric values).
    """
    flat: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_for_metrics(v, prefix=key))
        elif isinstance(v, bool):
            flat[key] = float(v)
        elif isinstance(v, (int, float)):
            flat[key] = float(v)
    return flat
