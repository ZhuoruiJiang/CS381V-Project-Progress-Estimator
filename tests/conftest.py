from types import SimpleNamespace

import datasets
import pytest
import torch

CAMERA = "observation.images.test"


class FakeLeRobotDataset:
    """Small episodes with source-identifiable pixels and metadata; no video I/O."""

    fps = 20
    delta_timestamps = None
    image_transforms = None
    features = {CAMERA: {}, "action": {}, "stage": {}}
    meta = SimpleNamespace(camera_keys=[CAMERA])

    def __init__(self):
        self.calls = []
        self.hf_dataset = datasets.Dataset.from_dict(
            {
                "episode_index": [2] * 4 + [9] * 3,
                "frame_index": [0, 1, 2, 3, 0, 1, 2],
                "timestamp": [0, 0.05, 0.1, 0.15, 0, 0.05, 0.1],
            }
        )

    def __getitem__(self, index):
        self.calls.append(index)
        return {
            CAMERA: torch.full((3, 8, 10), index / 10, dtype=torch.float32),
            "action": torch.tensor([float(index)]),
            "stage": f"stage-{index}",
        }


@pytest.fixture
def base():
    return FakeLeRobotDataset()
