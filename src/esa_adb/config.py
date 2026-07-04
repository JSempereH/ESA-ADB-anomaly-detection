from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

_DT_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.strptime(value, _DT_FMT) if value else None


def _fmt_dt(value: Optional[datetime]) -> Optional[str]:
    return value.strftime(_DT_FMT) if value else None


@dataclass
class GridBounds:
    start: datetime
    end: datetime

    def to_dict(self) -> dict:
        return {"start": _fmt_dt(self.start), "end": _fmt_dt(self.end)}

    @classmethod
    def from_dict(cls, d: dict) -> "GridBounds":
        return cls(start=_parse_dt(d["start"]), end=_parse_dt(d["end"]))


@dataclass
class SplitBoundaries:
    train_start: datetime
    val_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def to_dict(self) -> dict:
        return {
            "train_start": _fmt_dt(self.train_start),
            "val_start": _fmt_dt(self.val_start),
            "train_end": _fmt_dt(self.train_end),
            "test_start": _fmt_dt(self.test_start),
            "test_end": _fmt_dt(self.test_end),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SplitBoundaries":
        return cls(
            train_start=_parse_dt(d["train_start"]),
            val_start=_parse_dt(d["val_start"]),
            train_end=_parse_dt(d["train_end"]),
            test_start=_parse_dt(d["test_start"]),
            test_end=_parse_dt(d["test_end"]),
        )


@dataclass
class MissionConfig:
    mission_number: int
    raw_data_dir: Path
    interim_dir: Path
    processed_dir: Path
    target_frequency_seconds: int
    channel_count: int
    target_channel_count: int
    chunk_freq: str = "1y"
    monotonic_groups: list[int] = field(default_factory=list)
    categorical_max_unique: int = 10
    val_months: int = 3
    grid: Optional[GridBounds] = None
    split: Optional[SplitBoundaries] = None

    @property
    def mission_dir_name(self) -> str:
        return f"Mission{self.mission_number}"

    def to_dict(self) -> dict:
        return {
            "mission_number": self.mission_number,
            "raw_data_dir": str(self.raw_data_dir),
            "interim_dir": str(self.interim_dir),
            "processed_dir": str(self.processed_dir),
            "target_frequency_seconds": self.target_frequency_seconds,
            "channel_count": self.channel_count,
            "target_channel_count": self.target_channel_count,
            "chunk_freq": self.chunk_freq,
            "monotonic_groups": self.monotonic_groups,
            "categorical_max_unique": self.categorical_max_unique,
            "val_months": self.val_months,
            "grid": self.grid.to_dict() if self.grid else None,
            "split": self.split.to_dict() if self.split else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MissionConfig":
        return cls(
            mission_number=d["mission_number"],
            raw_data_dir=Path(d["raw_data_dir"]),
            interim_dir=Path(d["interim_dir"]),
            processed_dir=Path(d["processed_dir"]),
            target_frequency_seconds=d["target_frequency_seconds"],
            channel_count=d["channel_count"],
            target_channel_count=d["target_channel_count"],
            chunk_freq=d.get("chunk_freq", "1y"),
            monotonic_groups=d.get("monotonic_groups", []),
            categorical_max_unique=d.get("categorical_max_unique", 10),
            val_months=d.get("val_months", 3),
            grid=GridBounds.from_dict(d["grid"]) if d.get("grid") else None,
            split=SplitBoundaries.from_dict(d["split"]) if d.get("split") else None,
        )


def load_mission_config(path: str | Path) -> MissionConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return MissionConfig.from_dict(d)


def save_mission_config(config: MissionConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
