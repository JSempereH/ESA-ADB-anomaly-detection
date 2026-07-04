from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from esa_adb.config import GridBounds, MissionConfig, save_mission_config
from esa_adb.io.anomalies import build_anomalies_table
from esa_adb.io.channels import ingest_all_channels, list_channel_pickles, load_channel_metadata
from esa_adb.preprocessing.grid import ceil_to_resolution, floor_to_resolution, grid_slice_for_chunk
from esa_adb.preprocessing.labeling import any_target_anomaly_expr, extract_labels_long
from esa_adb.preprocessing.resample import CarryState, resample_channel_chunk
from esa_adb.preprocessing.split import compute_split_boundaries, label_split
from esa_adb.preprocessing.standardize import compute_standardization_stats
from esa_adb.preprocessing.typing_ import infer_channel_types


def _iter_chunks(start: datetime, end: datetime, chunk_freq: str):
    step = relativedelta(years=1) if chunk_freq == "1y" else relativedelta(months=1)
    cursor = datetime(start.year, 1, 1) if chunk_freq == "1y" else datetime(start.year, start.month, 1)
    while cursor <= end:
        chunk_start = max(start, cursor)
        next_cursor = cursor + step
        is_last = next_cursor > end
        chunk_end = end if is_last else next_cursor
        chunk_id = cursor.strftime("%Y") if chunk_freq == "1y" else cursor.strftime("%Y-%m")
        yield chunk_id, chunk_start, chunk_end, is_last
        if is_last:
            break
        cursor = next_cursor


def ingest_mission(
    config: MissionConfig, config_path: str | Path, force_reingest: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Channel metadata + anomalies table + per-channel raw Parquet ingestion.

    Also freezes the resampling grid bounds and train/val/test split boundaries into
    `config` (persisted to `config_path`) the first time it runs, so later resampling
    runs reuse fixed values instead of recomputing them. Returns
    (channel_meta, anomalies_df, target_channels) for reuse by the caller.
    """
    channel_names = [f"channel_{i}" for i in range(1, config.channel_count + 1)]

    channels_csv = Path(config.raw_data_dir) / "channels.csv"
    channel_meta = load_channel_metadata(channels_csv)
    if "channel" in channel_meta.columns:
        missing = set(channel_names) - set(channel_meta["channel"].to_list())
        if missing:
            raise ValueError(f"channels.csv is missing metadata for channels: {sorted(missing)[:5]}...")
    if "is_target" in channel_meta.columns:
        n_target = int(channel_meta["is_target"].sum())
        if n_target != config.target_channel_count:
            raise ValueError(
                f"Expected {config.target_channel_count} target channels per config, "
                f"found {n_target} in channels.csv"
            )
        target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()
    else:
        target_channels = []

    anomalies_path = Path(config.interim_dir) / f"anomalies_mission{config.mission_number}.csv"
    anomalies_df = build_anomalies_table(
        anomaly_types_path=Path(config.raw_data_dir) / "anomaly_types.csv",
        labels_path=Path(config.raw_data_dir) / "labels.csv",
        output_path=anomalies_path,
    )

    channels_raw_dir = Path(config.interim_dir) / "channels_raw"
    ingest_meta_path = Path(config.interim_dir) / "channel_ingest_meta.parquet"
    if force_reingest or not ingest_meta_path.exists():
        channel_pkls_dir = Path(config.raw_data_dir) / "channels"
        if not list_channel_pickles(channel_pkls_dir):
            from esa_adb.io.channels import extract_pkls_from_zips

            channel_pkls_dir = extract_pkls_from_zips(channel_pkls_dir)
        meta = ingest_all_channels(channel_pkls_dir, channel_names, anomalies_df, channels_raw_dir)
    else:
        meta = pl.read_parquet(ingest_meta_path)

    if config.grid is None:
        global_start = floor_to_resolution(meta["t_min"].min(), config.target_frequency_seconds)
        global_end = ceil_to_resolution(meta["t_max"].max(), config.target_frequency_seconds)
        config.grid = GridBounds(start=global_start, end=global_end)
        save_mission_config(config, config_path)

    if config.split is None:
        config.split = compute_split_boundaries(config.grid.start, config.grid.end, config.val_months)
        save_mission_config(config, config_path)

    return channel_meta, anomalies_df, target_channels


def run_mission_pipeline(config: MissionConfig, config_path: str | Path, force_reingest: bool = False) -> None:
    channel_names = [f"channel_{i}" for i in range(1, config.channel_count + 1)]
    channel_meta, _anomalies_df, target_channels = ingest_mission(config, config_path, force_reingest)

    global_start, global_end = config.grid.start, config.grid.end
    split_boundaries = config.split
    channels_raw_dir = Path(config.interim_dir) / "channels_raw"

    processed_dir = Path(config.processed_dir)
    chunks_dir = processed_dir / "chunks"
    labels_long_dir = processed_dir / "labels_long"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    labels_long_dir.mkdir(parents=True, exist_ok=True)

    carry_state: dict[str, CarryState] = {ch: CarryState() for ch in channel_names}

    for chunk_id, chunk_start, chunk_end, is_last in tqdm(
        list(_iter_chunks(global_start, global_end, config.chunk_freq)), desc=f"Mission{config.mission_number} chunks"
    ):
        grid = grid_slice_for_chunk(
            global_start, chunk_start, chunk_end, config.target_frequency_seconds, include_end=is_last
        )
        if grid.height == 0:
            continue

        time_filter = (
            (pl.col("time") >= chunk_start) & (pl.col("time") <= chunk_end)
            if is_last
            else (pl.col("time") >= chunk_start) & (pl.col("time") < chunk_end)
        )

        labels_long_frames = []
        wide = grid.clone()
        label_cols = []
        for ch in channel_names:
            raw_chunk = pl.scan_parquet(channels_raw_dir / f"{ch}.parquet").filter(time_filter).collect()

            resampled, new_carry = resample_channel_chunk(raw_chunk, grid, carry_state[ch], ch)
            carry_state[ch] = new_carry

            label_col = f"{ch}__label"
            labels_long_frames.append(extract_labels_long(resampled.rename({label_col: "label_code"}), ch))

            # Every per-channel frame is resampled onto the exact same `grid`, in the
            # same row order, so columns can be attached positionally -- no join needed.
            wide = wide.with_columns(resampled[ch], resampled[label_col])
            label_cols.append(label_col)

        target_label_cols = [f"{ch}__label" for ch in target_channels] if target_channels else []
        wide = wide.with_columns(any_target_anomaly_expr(target_label_cols))
        wide = wide.drop(label_cols)
        wide = wide.with_columns(label_split("time", split_boundaries))

        wide.write_parquet(chunks_dir / f"mission{config.mission_number}_{chunk_id}.parquet")

        if labels_long_frames:
            pl.concat(labels_long_frames, how="vertical").write_parquet(labels_long_dir / f"{chunk_id}.parquet")

    train_lf = pl.scan_parquet(chunks_dir / "*.parquet").filter(pl.col("split") == "train")
    channel_types = infer_channel_types(train_lf, channel_meta, config, channel_names)
    channel_types.write_parquet(processed_dir / "channel_types.parquet")

    labels_long_files = sorted(labels_long_dir.glob("*.parquet"))
    labels_long = (
        pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
        if labels_long_files
        else pl.DataFrame({"time": [], "channel": [], "category_code": []})
    )

    stats, categorical_mappings = compute_standardization_stats(train_lf, labels_long, channel_types, channel_names)
    stats.write_parquet(processed_dir / "standardization_stats.parquet")

    import json

    with open(processed_dir / "categorical_mappings.json", "w") as f:
        json.dump({k: [str(v) for v in vs] for k, vs in categorical_mappings.items()}, f)
