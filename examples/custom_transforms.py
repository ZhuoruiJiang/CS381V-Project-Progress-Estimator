"""Import this module before constructing a dataset to enable these examples."""

import random
from dataclasses import dataclass

import torch

from coffee_progress.augmentation import Timeline, register_augmentation
from coffee_progress.preprocessing import register_preprocessing


@register_preprocessing("clamp_pixels")
@dataclass
class ClampPixels:
    """Example: configure as {name: clamp_pixels, minimum: 0, maximum: 1}."""

    minimum: float = 0
    maximum: float = 1

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.clamp(self.minimum, self.maximum)


@register_augmentation("hold_start")
@dataclass
class HoldStart:
    """Example: configure as {name: hold_start, frames: 10}."""

    frames: int = 10

    def __post_init__(self):
        if self.frames < 1:
            raise ValueError("frames must be positive")

    def __call__(self, timeline: Timeline, rng: random.Random) -> Timeline:
        return timeline.insert(1, (timeline.source_frames[0],) * self.frames, "hold_start")
