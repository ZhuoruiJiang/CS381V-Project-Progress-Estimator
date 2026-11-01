import json
from dataclasses import replace

import datasets
import pytest

from coffee_progress import DatasetConfig, ProgressDataset, SourceConfig
from coffee_progress.splits import create_splits, load_splits


@pytest.fixture
def split_data(tmp_path, base):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 3,
                "total_frames": 9,
                "fps": 20,
            }
        )
    )
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"episode_index": 2, "length": 4},
                {"episode_index": 9, "length": 3},
                {"episode_index": 17, "length": 2},
            ]
        )
    )
    base.hf_dataset = datasets.concatenate_datasets(
        [
            base.hf_dataset,
            datasets.Dataset.from_dict(
                {
                    "episode_index": [17, 17],
                    "frame_index": [0, 1],
                    "timestamp": [0, 0.05],
                }
            ),
        ]
    )
    output = tmp_path / "splits.json"
    manifest = create_splits(root, output, ratios=(0.34, 0.34, 0.32))
    return root, output, manifest, base


def test_saved_splits_are_deterministic_complete_and_disjoint(split_data, tmp_path):
    root, output, manifest, _ = split_data
    again = create_splits(root, tmp_path / "again.json", ratios=(0.34, 0.34, 0.32))
    assert manifest == again == load_splits(output, root)
    groups = [set(ids) for ids in manifest["episodes"].values()]
    assert set.union(*groups) == {2, 9, 17}
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert set(manifest["qualitative_episodes"]) <= groups[2]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        create_splits(root, output, seed=100)
    assert output.read_bytes() == original


@pytest.mark.parametrize("mutation", ["overlap", "missing", "unknown", "qualitative", "release"])
def test_bad_manifests_are_rejected(split_data, mutation):
    root, output, manifest, _ = split_data
    if mutation == "overlap":
        manifest["episodes"]["val"] = manifest["episodes"]["train"]
    elif mutation == "missing":
        manifest["episodes"]["val"] = []
    elif mutation == "unknown":
        manifest["episodes"]["val"] = [999]
    elif mutation == "qualitative":
        manifest["qualitative_episodes"] = manifest["episodes"]["train"]
    else:
        manifest["dataset"]["metadata_sha256"] = "another-release"
    output.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_splits(output, root)


def test_split_selection_happens_before_augmentation_and_eval_is_clean(split_data):
    root, output, manifest, base = split_data
    config = DatasetConfig(
        source=SourceConfig(root=str(root), split="train", split_file=str(output)),
        augmentation=[{"name": "freeze", "min_frames": 2, "max_frames": 2}],
    )
    for name in ("train", "val", "test"):
        ds = ProgressDataset(config, split=name, base=base)
        assert list(ds.episode_ids) == manifest["episodes"][name]
        for episode_id in ds.episode_ids:
            original_length = len(ds.episodes[episode_id].source_indices)
            assert ds.episode_length(episode_id) == original_length + (2 if name == "train" else 0)
            sample = ds.get_history(episode_id, ds.episode_length(episode_id) - 1)
            assert bool(sample["events"]) == (name == "train")
        other_episode = manifest["episodes"]["val" if name == "train" else "train"][0]
        with pytest.raises(KeyError):
            ds.get_frame(other_episode, 0)


def test_explicit_episode_selection_cannot_override_split(split_data):
    root, output, _, base = split_data
    with pytest.raises(ValueError, match="not both"):
        SourceConfig(episodes=[2], split="train")
    source = SourceConfig(root=str(root), split_file=str(output), episodes=[2], split=None)
    ds = ProgressDataset(DatasetConfig(source=source), base=base)
    assert ds.episode_ids == (2,)
    with pytest.raises(ValueError, match="not both"):
        ProgressDataset(DatasetConfig(source=source), split="test", base=base)
    with pytest.raises(ValueError, match="split must"):
        replace(source, episodes=None, split="typo")


def test_missing_split_file_is_not_regenerated(split_data, tmp_path):
    root, _, _, base = split_data
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Restore"):
        ProgressDataset(
            DatasetConfig(source=SourceConfig(root=str(root), split="train", split_file=str(path))),
            base=base,
        )
    assert not path.exists()


@pytest.mark.parametrize("ratios", [(1, 0, 0), (0.8, 0.2, 0.2), (float("nan"), 0.1, 0.1)])
def test_invalid_ratios_are_rejected(split_data, tmp_path, ratios):
    with pytest.raises(ValueError):
        create_splits(split_data[0], tmp_path / "invalid.json", ratios=ratios)
