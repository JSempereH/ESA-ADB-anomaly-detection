#!/usr/bin/env python3
"""Run the full chunked ESA-ADB preprocessing pipeline for one mission.

Usage: python scripts/run_preprocessing.py --mission {1,2} [--force-reingest]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.config import load_mission_config
from esa_adb.pipeline import run_mission_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESA-ADB preprocessing pipeline")
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--force-reingest", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)

    run_mission_pipeline(config, config_path, force_reingest=args.force_reingest)
    print(f"Done. Processed chunks in: {Path(config.processed_dir) / 'chunks'}")


if __name__ == "__main__":
    main()
