"""A thin PyTorch wrapper around RoboCasa's upstream LeRobotDataset reader."""

import random
from bisect import bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .augmentation import AUGMENTATION, Timeline
from .config import AccessConfig, DatasetConfig
from .preprocessing import PREPROCESSING
from .splits import load_splits


@dataclass(frozen=True)
class Episode:
    source_indices: tuple[int, ...]
    timestamps: tuple[float, ...]


class ProgressDataset(Dataset):
    """Index frames on a virtual augmented timeline, without copying videos.

    ``dataset[i]`` uses the configured frame/history access mode and a flat index.
    ``get_frame(episode_id, t)`` and ``get_history(episode_id, t)`` use episode-local
    augmented indices. All images retain a time dimension, including single frames.

    ``base`` is an optional already constructed LeRobotDataset, useful for tests or
    team code that manages the upstream reader itself. Do not set delta_timestamps
    or image_transforms on that reader: this wrapper owns temporal access/transforms.
    """

    def __init__(self, config: DatasetConfig | None = None, *, base=None, split: str | None = None):
        self.config = config or DatasetConfig()
        if split is not None:
            self.config = replace(self.config, source=replace(self.config.source, split=split))
        source = self.config.source
        self.split = source.split
        self.split_manifest = (
            load_splits(source.split_file, source.root) if self.split is not None else None
        )
        self._selected_episodes = (
            self.split_manifest["episodes"][self.split]
            if self.split_manifest is not None
            else source.episodes
        )
        self.preprocessing = PREPROCESSING.build(self.config.preprocessing)
        self.augmentations = (
            [] if self.split in ("val", "test") else AUGMENTATION.build(self.config.augmentation)
        )
        self.base = base if base is not None else self._open_source()
        if getattr(self.base, "delta_timestamps", None):
            raise ValueError("Pass a base reader without delta_timestamps")
        if getattr(self.base, "image_transforms", None):
            raise ValueError("Pass a base reader without image_transforms")
        self.fps = float(self.base.fps)
        if self.fps <= 0:
            raise ValueError("Dataset fps must be positive")
        self.cameras = tuple(
            self.base.meta.camera_keys if self.config.cameras is None else self.config.cameras
        )
        if not self.cameras:
            raise ValueError("Choose at least one camera")
        unknown = set(self.cameras) - set(self.base.meta.camera_keys)
        if unknown:
            raise ValueError(f"Unknown cameras: {sorted(unknown)}")
        missing = set(self.config.metadata_fields) - set(self.base.features)
        if missing:
            raise ValueError(f"Metadata fields absent from dataset: {sorted(missing)}")
        self.episodes = self._index_episodes()
        self.episode_ids = tuple(sorted(self.episodes))
        self.timelines = {}
        self._ends = []
        total = 0
        for episode_id in self.episode_ids:
            timeline = Timeline.original(len(self.episodes[episode_id].source_indices))
            digest = sha256(f"{self.config.seed}:{episode_id}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "little"))
            for augment in self.augmentations:
                timeline = augment(timeline, rng)
                self._validate_timeline(timeline, len(self.episodes[episode_id].source_indices))
            self.timelines[episode_id] = timeline
            total += len(timeline.source_frames)
            self._ends.append(total)

    def _open_source(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        source = self.config.source
        root = Path(source.root)
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Dataset missing at {root}. Run coffee-download first.")
        return LeRobotDataset(
            repo_id=source.repo_id,
            root=root,
            episodes=self._selected_episodes,
            video_backend=source.video_backend,
        )

    def _index_episodes(self) -> dict[int, Episode]:
        # Read only small tabular columns; never decode videos to build the index.
        columns = (
            self.base.hf_dataset.select_columns(["episode_index", "frame_index", "timestamp"])
            .with_format(None)
            .to_dict()
        )
        groups: dict[int, list] = {}
        for source_index, (episode_id, frame, timestamp) in enumerate(
            zip(columns["episode_index"], columns["frame_index"], columns["timestamp"], strict=True)
        ):
            groups.setdefault(int(episode_id), []).append(
                (int(frame), source_index, float(timestamp))
            )
        if not groups:
            raise ValueError("Dataset contains no episodes")
        if self._selected_episodes is not None:
            missing = set(self._selected_episodes) - set(groups)
            if missing:
                raise ValueError(f"Selected episodes absent from the reader: {sorted(missing)}")
            groups = {i: groups[i] for i in self._selected_episodes}
        result = {}
        for episode_id, rows in groups.items():
            rows.sort()
            if [row[0] for row in rows] != list(range(len(rows))):
                raise ValueError(f"Episode {episode_id} has missing or duplicate source frames")
            result[episode_id] = Episode(
                tuple(row[1] for row in rows), tuple(row[2] for row in rows)
            )
        return result

    @staticmethod
    def _validate_timeline(timeline: Timeline, source_length: int):
        if not timeline.source_frames or len(timeline.source_frames) != len(timeline.kinds):
            raise ValueError("Augmentation must return nonempty, equally sized source_frames/kinds")
        if any(i < 0 or i >= source_length for i in timeline.source_frames):
            raise ValueError("Augmentation returned an out-of-range source frame")

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect_right(self._ends, index)
        start = self._ends[episode_position - 1] if episode_position else 0
        return self._get(self.episode_ids[episode_position], index - start, self.config.access)

    def episode_length(self, episode_id: int) -> int:
        """Number of timesteps after configured temporal augmentation."""
        return len(self.timelines[episode_id].source_frames)

    def get_frame(self, episode_id: int, frame_index: int) -> dict:
        return self._get(episode_id, frame_index, AccessConfig(mode="frame"))

    def get_history(
        self,
        episode_id: int,
        frame_index: int,
        *,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> dict:
        """Return the causal prefix, optionally strided/capped, always ending at t."""
        return self._get(episode_id, frame_index, AccessConfig("history", stride, max_frames))

    def iter_episode_frames(self, episode_id: int, *, chunk_size: int = 32, image_transform=None):
        """Read an unaugmented episode in bounded chunks through the LeRobot decoder.

        image_transform optionally changes raw [0,1] RGB before preprocessing.
        All preprocessing still runs on one-frame clips, matching get_frame exactly.
        The pinned LeRobot 0.3.3 query method avoids re-opening videos for every frame.
        Custom readers without that method fall back to ordinary indexed access.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        episode = self.episodes[episode_id]
        if self.timelines[episode_id].source_frames != tuple(range(len(episode.source_indices))):
            raise ValueError("Sequential reading requires an unaugmented episode")
        if not hasattr(self.base, "_query_videos"):
            if image_transform is not None:
                raise ValueError("Raw image transforms require the upstream video reader")
            for index in range(len(episode.source_indices)):
                yield self.get_frame(episode_id, index)
            return
        for start in range(0, len(episode.source_indices), chunk_size):
            stop = min(start + chunk_size, len(episode.source_indices))
            times = list(episode.timestamps[start:stop])
            decoded = self.base._query_videos(
                {camera: times for camera in self.cameras}, episode_id
            )
            decoded = {c: f.unsqueeze(0) if f.ndim == 3 else f for c, f in decoded.items()}
            if image_transform is not None:
                decoded = {c: image_transform(f) for c, f in decoded.items()}
            indices = list(episode.source_indices[start:stop])
            columns = (
                self.base.hf_dataset.select_columns(self.config.metadata_fields)[indices]
                if self.config.metadata_fields
                else {}
            )
            rows = {
                source_index: {
                    **{c: decoded[c][offset] for c in self.cameras},
                    **{key: columns[key][offset] for key in self.config.metadata_fields},
                }
                for offset, source_index in enumerate(indices)
            }
            for index in range(start, stop):
                yield self._get(episode_id, index, AccessConfig("frame"), rows=rows)

    def _get(self, episode_id: int, frame_index: int, access: AccessConfig, *, rows=None) -> dict:
        timeline = self.timelines[episode_id]
        if frame_index < 0 or frame_index >= len(timeline.source_frames):
            raise IndexError(f"Episode {episode_id}: frame {frame_index} is out of range")
        positions = [frame_index]
        if access.mode == "history":
            positions = list(range(frame_index, -1, -access.stride))
            if access.max_frames is not None:
                positions = positions[: access.max_frames]
            positions.reverse()
        source_frames = [timeline.source_frames[t] for t in positions]
        episode = self.episodes[episode_id]
        source_indices = [episode.source_indices[t] for t in source_frames]
        # Repeated frozen/rewound frames are decoded once per sample. Upstream
        # LeRobot handles MP4 seeking and synchronization; no second video reader.
        if rows is None:
            rows = {i: self.base[i] for i in dict.fromkeys(source_indices)}
        images = {}
        for camera in self.cameras:
            frames = torch.stack([rows[i][camera] for i in source_indices])
            if frames.ndim != 4 or frames.shape[1] != 3:
                raise ValueError(f"Camera {camera} must provide RGB [C,H,W] frames")
            for preprocess in self.preprocessing:
                frames = preprocess(frames)
                if frames.ndim != 4 or frames.shape[0] != len(positions):
                    raise ValueError("Preprocessing must preserve the clip's time dimension")
            images[camera] = frames
        events = []
        for event in timeline.events():
            # Expose only the observed portion of each augmentation event.
            if event["start"] <= frame_index:
                event = {**event, "end": min(event["end"], frame_index + 1)}
                event["start_time"] = event["start"] / self.fps
                event["end_time"] = event["end"] / self.fps
                events.append(event)
        return {
            "images": images,
            "episode_id": episode_id,
            "frame_index": frame_index,
            "frame_indices": torch.tensor(positions, dtype=torch.long),
            "source_frame_indices": torch.tensor(source_frames, dtype=torch.long),
            "source_indices": torch.tensor(source_indices, dtype=torch.long),
            "timestamps": torch.tensor(positions, dtype=torch.float64) / self.fps,
            "source_timestamps": torch.tensor(
                [episode.timestamps[t] for t in source_frames], dtype=torch.float64
            ),
            "augmentation_kind": [timeline.kinds[t] for t in positions],
            "events": events,
            "metadata": {
                key: [rows[i][key] for i in source_indices] for key in self.config.metadata_fields
            },
        }
