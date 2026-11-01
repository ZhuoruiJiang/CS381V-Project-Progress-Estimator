"""Episode-level temporal transforms; edit frame indices instead of video files."""

import random
from dataclasses import dataclass

from .registry import Registry

AUGMENTATION = Registry("augmentation method")
register_augmentation = AUGMENTATION.register


@dataclass(frozen=True)
class Timeline:
    """Every output position points to an original episode-local source frame."""

    source_frames: tuple[int, ...]
    kinds: tuple[str, ...]

    @classmethod
    def original(cls, length: int):
        return cls(tuple(range(length)), ("original",) * length)

    def insert(self, position: int, source_frames: tuple[int, ...], kind: str):
        return Timeline(
            self.source_frames[:position] + source_frames + self.source_frames[position:],
            self.kinds[:position] + (kind,) * len(source_frames) + self.kinds[position:],
        )

    def events(self) -> list[dict]:
        """Intervals use augmented episode positions; end is exclusive."""
        result = []
        start = 0
        while start < len(self.kinds):
            kind = self.kinds[start]
            end = start + 1
            while end < len(self.kinds) and self.kinds[end] == kind:
                end += 1
            if kind != "original":
                result.append({"kind": kind, "start": start, "end": end})
            start = end
        return result


@dataclass
class TemporalAugmentation:
    probability: float = 1.0
    min_frames: int = 5
    max_frames: int = 20

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0,1]")
        if self.min_frames < 1 or self.max_frames < self.min_frames:
            raise ValueError("Require 1 <= min_frames <= max_frames")


@register_augmentation("rewind")
@dataclass
class Rewind(TemporalAugmentation):
    """Insert the reversed copy of a randomly selected contiguous clip."""

    def __call__(self, timeline: Timeline, rng: random.Random) -> Timeline:
        if rng.random() >= self.probability or len(timeline.source_frames) < self.min_frames:
            return timeline
        length = rng.randint(self.min_frames, min(self.max_frames, len(timeline.source_frames)))
        start = rng.randint(0, len(timeline.source_frames) - length)
        end = start + length
        return timeline.insert(end, timeline.source_frames[start:end][::-1], "rewind")


@register_augmentation("freeze")
@dataclass
class Freeze(TemporalAugmentation):
    """Repeat a random frame for min_frames..max_frames additional timesteps."""

    def __call__(self, timeline: Timeline, rng: random.Random) -> Timeline:
        if rng.random() >= self.probability:
            return timeline
        anchor = rng.randrange(len(timeline.source_frames))
        length = rng.randint(self.min_frames, self.max_frames)
        return timeline.insert(anchor + 1, (timeline.source_frames[anchor],) * length, "freeze")
