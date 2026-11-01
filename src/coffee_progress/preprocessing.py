"""Deterministic clip transforms. Each callable accepts and returns [T, C, H, W]."""

from dataclasses import dataclass

import torch
from torchvision.transforms import functional as F

from .registry import Registry

PREPROCESSING = Registry("preprocessing method")
register_preprocessing = PREPROCESSING.register


@register_preprocessing("resize")
@dataclass
class Resize:
    size: tuple[int, int] = (224, 224)

    def __post_init__(self):
        if len(self.size) != 2 or min(self.size) < 1:
            raise ValueError("resize.size must contain two positive dimensions")

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return F.resize(frames, list(self.size), antialias=True)


@register_preprocessing("center_crop")
@dataclass
class CenterCrop:
    size: tuple[int, int] = (224, 224)

    def __post_init__(self):
        if len(self.size) != 2 or min(self.size) < 1:
            raise ValueError("center_crop.size must contain two positive dimensions")

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return F.center_crop(frames, list(self.size))


@register_preprocessing("to_float")
@dataclass
class ToFloat:
    """Convert uint8 [0,255] to float32 [0,1]; preserve floating point scale."""

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype == torch.uint8:
            return frames.float().div(255)
        if not frames.is_floating_point():
            raise TypeError("to_float expects uint8 or floating point images")
        return frames.float()


@register_preprocessing("normalize")
@dataclass
class Normalize:
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if len(self.mean) != len(self.std) or not self.std or min(self.std) <= 0:
            raise ValueError("normalize requires matching mean/std and positive std")

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if not frames.is_floating_point():
            raise TypeError("Run to_float before normalize")
        return F.normalize(frames, list(self.mean), list(self.std))
