#!/usr/bin/env python3
"""Ingest raw ESA-ADB pickles into per-channel Parquet and freeze grid/split config.

Usage: python scripts/ingest_raw.py --mission {1,2} [--force-reingest]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esa_adb.config import load_mission_config
from esa_adb.pipeline import ingest_mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw ESA-ADB mission data")
    parser.add_argument("--mission", type=int, required=True, choices=[1, 2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--force-reingest", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)

    channel_meta, anomalies_df, target_channels = ingest_mission(
        config, config_path, force_reingest=args.force_reingest
    )
    print(f"Ingested {len(channel_meta)} channels, {anomalies_df.height} annotated segments, "
          f"{len(target_channels)} target channels.")
    print(f"Grid: {config.grid.start} -> {config.grid.end}")
    print(f"Split: train [{config.split.train_start}, {config.split.val_start}) "
          f"val [{config.split.val_start}, {config.split.train_end}) "
          f"test [{config.split.test_start}, {config.split.test_end}]")


if __name__ == "__main__":
    main()
