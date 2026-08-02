from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Literal

from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward step: train on [train_start, train_end), then tune the
    threshold AND report final metrics on [eval_start, eval_end) -- eval_start ==
    train_end, i.e. the evaluation period is exactly the next step of unseen data, per
    the user's chosen semantics (not a separate third held-out period).
    """

    fold_index: int
    train_start: datetime
    train_end: datetime
    eval_start: datetime
    eval_end: datetime


def generate_walk_forward_folds(
    start: datetime,
    end: datetime,
    step_months: int = 1,
    window: Literal["expanding", "sliding"] = "expanding",
    sliding_window_months: int = 1,
    step_weeks: int | None = None,
    sliding_window_weeks: int | None = None,
) -> Iterator[WalkForwardFold]:
    """Monthly (or `step_months`-granularity) walk-forward folds over [start, end).

    `window="expanding"`: `train_start` stays fixed at `start` -- each fold trains on
    the *entire* history seen so far (a "stable" learner in the paired-learners sense).
    `window="sliding"`: `train_start` trails `train_end` by `sliding_window_months`,
    discarding everything older (a "reactive" learner) -- adapts faster to drift but
    with much less data per fold, and noisier as a result.

    `step_weeks` (and, for sliding windows, `sliding_window_weeks`) switch the fold
    granularity to weeks and take precedence over the month-based parameters -- the
    retraining-cadence axis (weekly / bi-weekly / monthly / bi-monthly) is itself an
    experimental variable in this project, not a fixed constant.

    The first fold needs one full step of history before its first evaluation period
    exists, so folds start at `start + step`.
    """
    step = relativedelta(weeks=step_weeks) if step_weeks else relativedelta(months=step_months)
    if window == "sliding":
        trail = relativedelta(weeks=sliding_window_weeks) if sliding_window_weeks else relativedelta(months=sliding_window_months)
        if step_weeks and sliding_window_weeks is None:
            trail = relativedelta(weeks=step_weeks)
    cursor = start + step
    fold_index = 0
    while cursor + step <= end:
        train_end = cursor
        train_start = start if window == "expanding" else max(start, train_end - trail)
        eval_end = cursor + step
        yield WalkForwardFold(fold_index, train_start, train_end, train_end, eval_end)
        cursor = eval_end
        fold_index += 1


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    """Micro-average (sum confusion-matrix counts across folds, then compute
    precision/recall/F-beta once) plus a macro-average (mean of each fold's own F-beta)
    for comparison -- micro is the statistically correct way to combine folds of very
    different sizes; macro treats every fold equally regardless of how much data it had,
    which can be misleading if some folds are much smaller than others.
    """
    if not fold_results:
        return {"micro": {}, "macro": {}, "n_folds": 0}

    tp = sum(r["tp"] for r in fold_results)
    fp = sum(r["fp"] for r in fold_results)
    fn = sum(r["fn"] for r in fold_results)
    beta = fold_results[0].get("beta", 0.5)
    beta2 = beta**2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0

    macro_f_beta = sum(r["f_beta"] for r in fold_results) / len(fold_results)

    return {
        "micro": {"precision": precision, "recall": recall, "f_beta": f_beta, "tp": tp, "fp": fp, "fn": fn},
        "macro": {"f_beta": macro_f_beta},
        "n_folds": len(fold_results),
    }
