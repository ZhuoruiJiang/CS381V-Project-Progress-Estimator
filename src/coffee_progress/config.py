"""Small configuration objects; YAML uses the same names as the Python API."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .splits import DEFAULT_SPLIT_FILE


@dataclass(frozen=True)
class SourceConfig:
    root: str = "data/prepare_coffee/lerobot"
    repo_id: str = "local/PrepareCoffee"
    episodes: list[int] | None = None
    video_backend: str = "pyav"
    split: Literal["train", "val", "test"] | None = "train"
    split_file: str = DEFAULT_SPLIT_FILE

    def __post_init__(self):
        if self.split is not None and self.split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, test, or null")
        if self.split is not None and self.episodes is not None:
            raise ValueError("Choose source.split or source.episodes, not both")


@dataclass(frozen=True)
class AccessConfig:
    mode: Literal["frame", "history"] = "frame"
    stride: int = 1
    max_frames: int | None = None

    def __post_init__(self):
        if self.mode not in ("frame", "history"):
            raise ValueError("mode must be 'frame' or 'history'")
        if self.stride < 1 or (self.max_frames is not None and self.max_frames < 1):
            raise ValueError("stride and max_frames must be positive")


@dataclass(frozen=True)
class DatasetConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    cameras: list[str] | None = None
    access: AccessConfig = field(default_factory=AccessConfig)
    preprocessing: list[dict] = field(default_factory=list)
    augmentation: list[dict] = field(default_factory=list)
    metadata_fields: list[str] = field(default_factory=list)
    seed: int = 0


def load_config(path: str | Path) -> DatasetConfig:
    """Paths in YAML are relative to the working directory, like the Python API."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    source = SourceConfig(**raw.pop("source", {}))
    access = AccessConfig(**raw.pop("access", {}))
    return DatasetConfig(source=source, access=access, **raw)
