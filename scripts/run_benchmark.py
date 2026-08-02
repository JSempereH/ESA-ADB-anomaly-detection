#!/usr/bin/env python3
"""Unified benchmark runner -- supersedes the per-experiment demo_* scripts for model
comparison (they remain for the analyses they uniquely own, e.g. explainability).

Both protocols share runners, threshold calibration and alarm post-processing
(see `esa_adb/benchmark.py`), so their corrected event-wise F0.5 numbers are directly
comparable -- to each other and to the ESA-ADB paper's published results.

Examples:
  python scripts/run_benchmark.py --mission 2 --protocol walk-forward --models GlobalSTD,PCAReconstructor
  python scripts/run_benchmark.py --mission 2 --protocol walk-forward --step-weeks 2
  python scripts/run_benchmark.py --mission 2 --protocol paper-split --calibration event
  python scripts/run_benchmark.py --mission 2 --protocol walk-forward --models Autoencoder-replay --variants online
  python scripts/run_benchmark.py --mission 2 --protocol walk-forward --full-history --models GlobalSTD,PCAReconstructor
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import DEFAULT_MODELS, load_benchmark_data, run_channel_aware, run_paper_split, run_walk_forward
from esa_adb.models.postprocess import PostprocessSpec


def _parse_alphas(raw: str) -> tuple:
    return tuple(None if part.strip().lower() in ("none", "") else float(part) for part in raw.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--protocol", choices=["walk-forward", "paper-split", "channel-aware"], default="walk-forward")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--variants", type=str, default="refit,online,hybrid", help="walk-forward only")
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--step-weeks", type=int, default=None, help="overrides --step-months")
    parser.add_argument("--calibration", choices=["event", "row", "budget"], default="event",
                        help="'budget' is the fully unsupervised production mode (no labels anywhere)")
    parser.add_argument("--alarm-budget", type=float, default=20.0, help="max postprocessed alarms per 30 days for --calibration budget")
    parser.add_argument("--max-alarm-time-frac", type=float, default=0.02, help="max fraction of the timeline inside an alarm for --calibration budget")
    parser.add_argument("--fit-on-all-rows", action="store_true",
                        help="skip the label-informed nominal filter before fitting (fully unsupervised training data)")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--merge-gap-rows", type=int, default=100)
    parser.add_argument("--min-duration-rows", type=int, default=10)
    parser.add_argument("--smooth-alphas", type=str, default="none,0.3")
    parser.add_argument("--exclude-channels", type=str, default=None, help="comma-separated channels to drop from the target set")
    parser.add_argument("--with-telecommands", action="store_true", help="append telecommand activity features as extra model inputs (build_telecommand_features.py first)")
    parser.add_argument("--ae-replay-rows", type=int, default=50_000, help="replay buffer for Autoencoder-replay")
    parser.add_argument("--ae-online-epochs", type=int, default=10, help="partial_fit epochs for Autoencoder-replay")
    parser.add_argument("--ae-online-lr", type=float, default=1e-3, help="partial_fit learning rate for Autoencoder-replay")
    parser.add_argument("--ae-seed", type=int, default=42, help="Autoencoder weight-init/shuffle seed (multi-seed statistical runs)")
    parser.add_argument("--warmup-folds", type=int, default=0, help="first N folds train/calibrate but are excluded from metrics (cold-start commissioning period)")
    parser.add_argument("--full-history", action="store_true",
                        help="walk-forward only: start folds at config.split.train_start (the mission's very first month) "
                             "instead of config.split.test_start -- the ENTIRE mission becomes one continuous walk-forward, "
                             "with no period held out as a separate paper-comparable split. Roughly doubles n_folds/runtime.")
    parser.add_argument("--ae-epochs", type=int, default=30, help="paper-split Autoencoder epochs")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    pp = PostprocessSpec(
        merge_gap_rows=args.merge_gap_rows,
        min_duration_rows=args.min_duration_rows,
        smooth_alphas=_parse_alphas(args.smooth_alphas),
    )
    exclude = args.exclude_channels.split(",") if args.exclude_channels else None
    data = load_benchmark_data(args.mission, args.config_dir, exclude_channels=exclude, with_telecommands=args.with_telecommands)
    model_names = [m for m in args.models.split(",") if m]

    if args.protocol == "walk-forward":
        run_walk_forward(
            data, model_names,
            variants=tuple(args.variants.split(",")),
            step_months=args.step_months, step_weeks=args.step_weeks,
            calibration=args.calibration, pp=pp, max_folds=args.max_folds,
            mlflow=args.mlflow, tag=args.tag, ae_replay_rows=args.ae_replay_rows,
            ae_online_epochs=args.ae_online_epochs, ae_online_lr=args.ae_online_lr,
            alarm_budget=args.alarm_budget, max_alarm_time_fraction=args.max_alarm_time_frac,
            fit_on_all_rows=args.fit_on_all_rows, warmup_folds=args.warmup_folds, ae_seed=args.ae_seed,
            full_history=args.full_history,
        )
    elif args.protocol == "channel-aware":
        run_channel_aware(
            data, model_names,
            step_months=args.step_months, calibration=args.calibration, pp=pp,
            alarm_budget=args.alarm_budget, max_alarm_time_fraction=args.max_alarm_time_frac,
            mlflow=args.mlflow, tag=args.tag,
        )
    else:
        run_paper_split(
            data, model_names,
            calibration=args.calibration, pp=pp,
            mlflow=args.mlflow, tag=args.tag, ae_refit_epochs=args.ae_epochs,
            alarm_budget=args.alarm_budget, max_alarm_time_fraction=args.max_alarm_time_frac,
        )


if __name__ == "__main__":
    main()
