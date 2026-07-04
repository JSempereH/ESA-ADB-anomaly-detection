from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import polars as pl
from tqdm import tqdm

from esa_adb.io.anomalies import attach_raw_labels


def has_pickle_files(directory: str | Path) -> bool:
    directory = Path(directory)
    return directory.is_dir() and any(f.suffix == ".pkl" for f in directory.iterdir())


def extract_pkls_from_zips(input_dir: str | Path, output_dir: Optional[str | Path] = None) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "extracted_pkls"
    output_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(input_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                out_file = output_dir / (Path(member).name + ".pkl")
                with zf.open(member) as src, open(out_file, "wb") as dst:
                    dst.write(src.read())
    return output_dir


def _channel_sort_key(filename: str) -> int:
    match = re.search(r"channel_(\d+)", filename)
    return int(match.group(1)) if match else 0


def list_channel_pickles(directory: str | Path) -> list[str]:
    directory = Path(directory)
    files = [f.name for f in directory.iterdir() if f.name.startswith("channel_") and f.name.endswith(".pkl")]
    return sorted(files, key=_channel_sort_key)


def load_channel_metadata(path: str | Path) -> pl.DataFrame:
    """Load channels.csv: channel name, subsystem, unit, group number, target/categorical flags.

    Column names are normalized defensively (case/whitespace). Confirmed against a real
    Mission2 download: ``Channel,Subsystem,Physical Unit,Group,Target,Categorical`` with
    YES/NO string values for the boolean columns - the ``Categorical`` flag isn't
    mentioned in the paper text but is present in the real release, and is more precise
    than the data-driven categorical heuristic in typing_.infer_channel_types, so it's
    used as a group-metadata-style override there when available.
    """
    df = pl.read_csv(str(path))
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in ("channel", "channel_name", "name"):
            rename[col] = "channel"
        elif key in ("group", "group_number", "group_no"):
            rename[col] = "group"
        elif key in ("target", "is_target", "target_channel"):
            rename[col] = "is_target"
        elif key in ("categorical", "is_categorical"):
            rename[col] = "is_categorical"
        elif key in ("subsystem",):
            rename[col] = "subsystem"
        elif key in ("unit", "physical_unit"):
            rename[col] = "unit"
    df = df.rename(rename)
    for bool_col in ("is_target", "is_categorical"):
        if bool_col in df.columns and df[bool_col].dtype != pl.Boolean:
            df = df.with_columns(
                pl.col(bool_col).cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "yes", "target"]).alias(bool_col)
            )
    return df


def _read_channel_pickle(file_path: str | Path, channel_name: str) -> pl.DataFrame:
    import pandas as pd

    series = pd.read_pickle(file_path)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    idx = pd.to_datetime(series.index)
    df = pl.DataFrame({"time": idx, "value": series.to_numpy()})

    if df.schema["value"] in (pl.Utf8, pl.String):
        # Categorical/status channels (channels.csv "Categorical" = YES) store values as
        # strings like "value_0", "value_1", ... rather than raw floats. Extract the
        # integer suffix; fall back to a direct numeric parse for any string that doesn't
        # match (e.g. plain numeric-looking strings).
        extracted = pl.col("value").str.extract(r"value_(\d+)$", 1).cast(pl.Float32, strict=False)
        direct = pl.col("value").cast(pl.Float32, strict=False)
        df = df.with_columns(pl.coalesce([extracted, direct]).alias("value"))
    else:
        df = df.with_columns(pl.col("value").cast(pl.Float32))

    # Match the (naive, microsecond) dtype produced by io.anomalies.build_anomalies_table
    # so later asof-joins (attach_raw_labels, resampling) have consistent key dtypes.
    df = df.with_columns(pl.col("time").cast(pl.Datetime("us")))
    return df.sort("time")


def ingest_all_channels(
    channel_pkls_dir: str | Path,
    channel_names: list[str],
    anomalies_df: pl.DataFrame,
    interim_channels_dir: str | Path,
) -> pl.DataFrame:
    """Read raw per-channel pickles once, attach raw-level labels, write per-channel Parquet.

    Sequential (not thread-pooled): keeps peak RAM to roughly one channel's raw series at
    a time, which matters for laptop-scale runs. Returns the ingest metadata table
    (channel, n_rows, t_min, t_max) used downstream to compute the global resampling grid
    without ever holding all channels in memory simultaneously.
    """
    channel_pkls_dir = Path(channel_pkls_dir)
    interim_channels_dir = Path(interim_channels_dir)
    interim_channels_dir.mkdir(parents=True, exist_ok=True)

    available = {f: f for f in os.listdir(channel_pkls_dir) if f.endswith(".pkl")}
    meta_rows = []

    for channel_name in tqdm(channel_names, desc="Ingesting channels"):
        pkl_name = f"{channel_name}.pkl"
        if pkl_name not in available:
            raise FileNotFoundError(f"Missing pickle for channel {channel_name!r} in {channel_pkls_dir}")

        raw_df = _read_channel_pickle(channel_pkls_dir / pkl_name, channel_name)
        raw_df = attach_raw_labels(raw_df, anomalies_df, channel_name)

        out_path = interim_channels_dir / f"{channel_name}.parquet"
        raw_df.write_parquet(out_path)

        meta_rows.append(
            {
                "channel": channel_name,
                "n_rows": raw_df.height,
                "t_min": raw_df["time"].min(),
                "t_max": raw_df["time"].max(),
            }
        )

    meta = pl.DataFrame(meta_rows)
    meta.write_parquet(interim_channels_dir.parent / "channel_ingest_meta.parquet")
    return meta
