"""Opt-in checks through the real LeRobot MP4 decoder and worker processes."""

import os
from pathlib import Path

import pytest
import torch

from coffee_progress import AccessConfig, DatasetConfig, ProgressDataset, SourceConfig
from coffee_progress.loader import make_dataloader
from coffee_progress.splits import DEFAULT_SPLIT_FILE, load_splits

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("COFFEE_INTEGRATION") != "1", reason="Set COFFEE_INTEGRATION=1"
    ),
]


def test_official_dataset_frame_history_and_workers():
    root = Path(os.environ.get("COFFEE_DATA_ROOT", "data/prepare_coffee/lerobot"))
    cfg = DatasetConfig(
        source=SourceConfig(root=str(root), episodes=[0, 2], split=None),
        access=AccessConfig("history", stride=2, max_frames=3),
        preprocessing=[{"name": "resize", "size": [32, 32]}],
        augmentation=[
            {"name": "rewind", "min_frames": 2, "max_frames": 2},
            {"name": "freeze", "min_frames": 2, "max_frames": 2},
        ],
        metadata_fields=["action", "observation.state", "subtask_idx"],
    )
    ds = ProgressDataset(cfg)
    assert ds.episode_ids == (0, 2)
    sample = ds.get_history(2, 8, stride=2, max_frames=3)
    assert sample["frame_indices"].tolist() == [4, 6, 8]
    for frames in sample["images"].values():
        assert frames.shape == (3, 3, 32, 32)
        assert torch.isfinite(frames).all()
    assert set(sample["metadata"]) == {"action", "observation.state", "subtask_idx"}
    freeze = next(e for e in ds.timelines[2].events() if e["kind"] == "freeze")
    frozen = ds.get_history(2, freeze["end"] - 1, max_frames=3)
    assert frozen["source_frame_indices"].unique().numel() == 1
    for frames in frozen["images"].values():
        torch.testing.assert_close(frames[0], frames[-1])
    # Construct a fresh reader before starting workers (upstream decoder guidance).
    worker_ds = ProgressDataset(cfg)
    batch = next(iter(make_dataloader(worker_ds, batch_size=3, num_workers=2)))
    assert batch["lengths"].tolist() == [1, 1, 2]
    assert batch["episode_id"].tolist() == [0, 0, 0]


def test_saved_official_splits_use_real_video_without_evaluation_augmentation():
    root = Path(os.environ.get("COFFEE_DATA_ROOT", "data/prepare_coffee/lerobot"))
    manifest = load_splits(DEFAULT_SPLIT_FILE, root)
    assert [len(manifest["episodes"][s]) for s in ("train", "val", "test")] == [411, 51, 52]
    cfg = DatasetConfig(
        source=SourceConfig(root=str(root), split="test"),
        cameras=["observation.images.robot0_agentview_left"],
        augmentation=[{"name": "freeze", "min_frames": 2, "max_frames": 2}],
    )
    ds = ProgressDataset(cfg)
    assert list(ds.episode_ids) == manifest["episodes"]["test"]
    sample = ds.get_frame(manifest["qualitative_episodes"][0], 1)
    assert sample["images"][cfg.cameras[0]].shape == (1, 3, 256, 256)
    assert not sample["events"]
    assert all(t.kinds.count("freeze") == 0 for t in ds.timelines.values())


def test_sequential_decoder_matches_indexed_access_for_all_cameras():
    root = Path(os.environ.get("COFFEE_DATA_ROOT", "data/prepare_coffee/lerobot"))
    ds = ProgressDataset(
        DatasetConfig(
            source=SourceConfig(root=str(root), episodes=[0], split=None),
            preprocessing=[{"name": "resize", "size": [32, 32]}],
            metadata_fields=["subtask_idx"],
        )
    )
    iterator = ds.iter_episode_frames(0, chunk_size=4)
    for index in range(6):
        sequential = next(iterator)
        indexed = ds.get_frame(0, index)
        assert len(sequential["images"]) == 3
        for camera in ds.cameras:
            torch.testing.assert_close(sequential["images"][camera], indexed["images"][camera])
        assert sequential["metadata"] == indexed["metadata"]
