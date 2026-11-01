"""Batch variable-length histories with explicit right-padding masks."""

import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def collate_samples(samples: list[dict]) -> dict:
    lengths = torch.tensor([len(sample["frame_indices"]) for sample in samples])
    batch = {
        "images": {
            camera: pad_sequence([s["images"][camera] for s in samples], batch_first=True)
            for camera in samples[0]["images"]
        },
        "lengths": lengths,
        "valid_mask": torch.arange(int(lengths.max()))[None, :] < lengths[:, None],
        "episode_id": torch.tensor([s["episode_id"] for s in samples]),
        "frame_index": torch.tensor([s["frame_index"] for s in samples]),
    }
    for key in ("frame_indices", "source_frame_indices", "source_indices"):
        batch[key] = pad_sequence([s[key] for s in samples], batch_first=True, padding_value=-1)
    for key in ("timestamps", "source_timestamps"):
        batch[key] = pad_sequence(
            [s[key] for s in samples], batch_first=True, padding_value=float("nan")
        )
    # Keep arbitrary user metadata and variable-length events as per-sample lists.
    for key in ("augmentation_kind", "events", "metadata"):
        batch[key] = [s[key] for s in samples]
    return batch


def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(
    dataset,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Works for frame and history modes. Start with num_workers=0 for debugging."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_seed_worker,
        collate_fn=collate_samples,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
