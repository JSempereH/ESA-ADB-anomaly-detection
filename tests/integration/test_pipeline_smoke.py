from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

from esa_adb.config import MissionConfig, save_mission_config
from esa_adb.pipeline import run_mission_pipeline

CHANNEL_NAMES = ["channel_1", "channel_2", "channel_3"]
MISSION_START = datetime(2020, 1, 1, 0, 0, 0)
MISSION_END = datetime(2020, 1, 1, 3, 0, 0)


def _build_fake_mission(raw_dir: Path) -> None:
    channels_dir = raw_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)

    n = 240  # one sample every 45s across the 3h span
    times = [MISSION_START + timedelta(seconds=45 * i) for i in range(n)]

    channel_1 = pd.Series([float(i) for i in range(n)], index=pd.DatetimeIndex(times))
    channel_1.to_pickle(channels_dir / "channel_1.pkl")

    channel_2 = pd.Series([float(i % 2) for i in range(n)], index=pd.DatetimeIndex(times))
    channel_2.to_pickle(channels_dir / "channel_2.pkl")

    channel_3 = pd.Series([10.0 + i * 0.1 for i in range(n)], index=pd.DatetimeIndex(times))
    channel_3.to_pickle(channels_dir / "channel_3.pkl")

    pd.DataFrame(
        {
            "channel": CHANNEL_NAMES,
            "subsystem": ["SYS1", "SYS1", "SYS2"],
            "unit": ["V", "bool", "V"],
            "group": [1, 1, 5],
            "is_target": [False, False, True],
        }
    ).to_csv(raw_dir / "channels.csv", index=False)

    pd.DataFrame(
        {
            "ID": [1, 2],
            "Channel": ["channel_3", "channel_3"],
            "StartTime": ["2020-01-01 01:00:00", "2020-01-01 02:00:00"],
            "EndTime": ["2020-01-01 01:05:00", "2020-01-01 02:05:00"],
        }
    ).to_csv(raw_dir / "labels.csv", index=False)

    pd.DataFrame(
        {
            "ID": [1, 2],
            "Class": ["SomeClass", "SomeClass"],
            "Subclass": ["SomeSub", "SomeSub"],
            "Category": ["Anomaly", "Rare Nominal Event"],
        }
    ).to_csv(raw_dir / "anomaly_types.csv", index=False)


def test_run_mission_pipeline_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    _build_fake_mission(raw_dir)

    config = MissionConfig(
        mission_number=1,
        raw_data_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        target_frequency_seconds=60,
        channel_count=3,
        target_channel_count=1,
        chunk_freq="1y",
        monotonic_groups=[],
        categorical_max_unique=5,
        val_months=0,
    )
    config_path = tmp_path / "mission1.yaml"
    save_mission_config(config, config_path)

    run_mission_pipeline(config, config_path)

    chunks_dir = processed_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    assert len(chunk_files) == 1

    df = pl.read_parquet(chunk_files[0])
    for ch in CHANNEL_NAMES:
        assert ch in df.columns
    assert "any_target_anomaly" in df.columns
    assert "split" in df.columns
    assert set(df["split"].unique().to_list()) <= {"train", "val", "test"}

    # 3h span at 60s resolution -> 181 grid points (both endpoints inclusive).
    assert df.height == 181
    assert df["time"].is_sorted()
    assert df["time"].n_unique() == df.height

    assert df["any_target_anomaly"].any()  # channel_3's anomaly window should surface here

    assert (processed_dir / "channel_types.parquet").exists()
    assert (processed_dir / "standardization_stats.parquet").exists()
    assert (processed_dir / "categorical_mappings.json").exists()

    channel_types = pl.read_parquet(processed_dir / "channel_types.parquet")
    assert set(channel_types["channel"].to_list()) == set(CHANNEL_NAMES)

    labels_long_files = list((processed_dir / "labels_long").glob("*.parquet"))
    assert labels_long_files
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files])
    assert set(labels_long["channel"].unique().to_list()) == {"channel_3"}

    with open(config_path) as f:
        reloaded = MissionConfig.from_dict(yaml.safe_load(f))
    assert reloaded.grid is not None
    assert reloaded.split is not None
