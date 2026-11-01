"""Create once, then reuse explicit episode splits across methods and machines."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_FILE = "configs/splits/prepare_coffee.json"


def _dataset_identity(root: str | Path) -> tuple[dict, list[int]]:
    """Identify a release from metadata, without opening videos or importing LeRobot."""
    meta = Path(root) / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = sorted((row["episode_index"], row["length"]) for row in episodes)
    ids = [episode_id for episode_id, _ in records]
    if len(ids) != len(set(ids)) or any(
        type(episode_id) is not int or episode_id < 0 or type(length) is not int or length < 1
        for episode_id, length in records
    ):
        raise ValueError("Dataset metadata contains invalid or duplicate episodes")
    if len(ids) != info["total_episodes"] or sum(n for _, n in records) != info["total_frames"]:
        raise ValueError("Episode metadata disagrees with dataset totals")
    payload = json.dumps({"info": info, "episodes": records}, sort_keys=True).encode()
    identity = {
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "num_episodes": len(ids),
        "num_frames": info["total_frames"],
    }
    return identity, ids


def create_splits(
    root: str | Path,
    output: str | Path = DEFAULT_SPLIT_FILE,
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Save disjoint episode IDs; refuse overwrites, even with the same seed.

    Train and validation counts round down; test receives the remainder.
    These are random episode splits, not scene-disjoint evaluation splits.
    """
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("Provide three positive, finite train/val/test ratios")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Split file already exists: {output}. Keep it fixed across methods.")
    identity, ids = _dataset_identity(root)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train, n_val = int(len(ids) * ratios[0]), int(len(ids) * ratios[1])
    if min(n_train, n_val, len(ids) - n_train - n_val) < 1:
        raise ValueError("Each split must contain at least one episode; adjust ratios or dataset")
    partitions = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }
    manifest = {
        "version": 1,
        "method": "random_episode",
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "dataset": identity,
        "episodes": partitions,
        "qualitative_episodes": sorted(
            rng.sample(partitions["test"], min(5, len(partitions["test"])))
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def load_splits(path: str | Path, root: str | Path) -> dict:
    """Reject overlap, omitted/unknown IDs, and split files for a different release."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}. Restore the shared split file.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity, ids = _dataset_identity(root)
    if manifest.get("version") != 1 or manifest.get("dataset") != identity:
        raise ValueError("Split file does not match this dataset release")
    partitions = manifest.get("episodes", {})
    if set(partitions) != set(SPLIT_NAMES):
        raise ValueError("Split file must contain train, val, and test")
    assigned = []
    for name in SPLIT_NAMES:
        episodes = partitions[name]
        if (
            not isinstance(episodes, list)
            or not episodes
            or any(type(i) is not int for i in episodes)
        ):
            raise ValueError(f"Invalid episode list in split '{name}'")
        assigned.extend(episodes)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(ids):
        raise ValueError("Split IDs overlap or do not exactly cover the source episodes")
    qualitative = manifest.get("qualitative_episodes", [])
    if (
        not isinstance(qualitative, list)
        or any(type(i) is not int for i in qualitative)
        or len(qualitative) != min(5, len(partitions["test"]))
        or len(qualitative) != len(set(qualitative))
        or not set(qualitative) <= set(partitions["test"])
    ):
        raise ValueError("Qualitative episodes must be unique held-out test episodes")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/prepare_coffee/lerobot")
    parser.add_argument("--output", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    args = parser.parse_args()
    manifest = create_splits(args.root, args.output, seed=args.seed, ratios=tuple(args.ratios))
    print(" | ".join(f"{name}: {len(ids)}" for name, ids in manifest["episodes"].items()))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
