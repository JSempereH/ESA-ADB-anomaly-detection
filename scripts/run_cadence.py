#!/usr/bin/env python3
"""Prequential ("test-then-train") retraining-cadence comparison -- fixed weekly
schedules vs. drift-triggered adaptive updates vs. never updating, for one model.
See `esa_adb.benchmark.run_adaptive_cadence` for the full methodology.

Examples:
  python scripts/run_cadence.py --mission 2 --model PCAReconstructor
  python scripts/run_cadence.py --mission 2 --model GlobalSTD --ks-threshold 0.2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.benchmark import load_benchmark_data, run_adaptive_cadence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model", type=str, default="PCAReconstructor")
    parser.add_argument("--strategies", type=str, default="every-1,every-4,every-8,adaptive,never")
    parser.add_argument("--step-weeks", type=int, default=1)
    parser.add_argument("--init-weeks", type=int, default=4)
    parser.add_argument("--calib-weeks", type=int, default=4)
    parser.add_argument("--ks-threshold", type=float, default=0.15)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    data = load_benchmark_data(args.mission, args.config_dir)
    run_adaptive_cadence(
        data,
        model_name=args.model,
        strategies=tuple(args.strategies.split(",")),
        step_weeks=args.step_weeks,
        init_weeks=args.init_weeks,
        calib_weeks=args.calib_weeks,
        ks_threshold=args.ks_threshold,
        mlflow=args.mlflow,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
