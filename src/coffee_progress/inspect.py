"""Load configured data and print a compact smoke-test summary."""

import argparse

from .config import load_config
from .dataset import ProgressDataset
from .loader import make_dataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prepare_coffee.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"))
    args = parser.parse_args()
    dataset = ProgressDataset(load_config(args.config), split=args.split)
    batch = next(iter(make_dataloader(dataset, batch_size=2, num_workers=args.num_workers)))
    print(f"Episodes: {len(dataset.episode_ids)} | augmented frames: {len(dataset)}")
    print(f"Split: {dataset.split or 'manual/all'}")
    print(f"Source fields: {sorted(dataset.base.features)}")
    for camera, frames in batch["images"].items():
        print(f"{camera}: {tuple(frames.shape)} {frames.dtype}")
    print(f"Valid lengths: {batch['lengths'].tolist()}")


if __name__ == "__main__":
    main()
