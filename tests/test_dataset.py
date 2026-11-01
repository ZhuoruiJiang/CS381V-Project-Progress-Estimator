from dataclasses import replace

import pytest
import torch
from conftest import CAMERA

from coffee_progress import (
    AccessConfig,
    DatasetConfig,
    ProgressDataset,
    SourceConfig,
    make_dataloader,
)
from coffee_progress.augmentation import Timeline, register_augmentation
from coffee_progress.config import load_config
from coffee_progress.preprocessing import register_preprocessing


def manual_config(**kwargs):
    return DatasetConfig(source=SourceConfig(split=None), **kwargs)


def test_frame_lookup_and_episode_boundary(base):
    ds = ProgressDataset(manual_config(), base=base)
    assert len(ds) == 7
    assert ds.episode_ids == (2, 9)
    assert ds[3]["episode_id"] == 2
    assert ds[4]["episode_id"] == 9
    assert ds[4]["source_frame_indices"].tolist() == [0]
    assert ds.get_frame(9, 2)["images"][CAMERA].shape == (1, 3, 8, 10)
    assert base.calls == [3, 4, 4, 6]


def test_history_is_causal_and_always_includes_requested_endpoint(base):
    ds = ProgressDataset(manual_config(), base=base)
    assert ds.get_history(2, 3)["frame_indices"].tolist() == [0, 1, 2, 3]
    assert ds.get_history(2, 3, stride=2)["frame_indices"].tolist() == [1, 3]
    assert ds.get_history(2, 3, stride=2, max_frames=1)["frame_indices"].tolist() == [3]
    assert ds.get_history(9, 0)["source_indices"].tolist() == [4]


def test_rewind_preserves_pixels_timestamps_and_metadata(base):
    config = manual_config(
        augmentation=[{"name": "rewind", "min_frames": 4, "max_frames": 4}],
        metadata_fields=["action", "stage"],
    )
    ds = ProgressDataset(config, base=base)
    sample = ds.get_history(2, 7)
    source = [0, 1, 2, 3, 3, 2, 1, 0]
    assert sample["source_frame_indices"].tolist() == source
    assert sample["images"][CAMERA][:, 0, 0, 0].tolist() == pytest.approx([i / 10 for i in source])
    assert sample["source_timestamps"].tolist() == pytest.approx([i / 20 for i in source])
    assert sample["timestamps"].tolist() == pytest.approx([i / 20 for i in range(8)])
    assert sample["metadata"]["stage"] == [f"stage-{i}" for i in source]
    assert [x.item() for x in sample["metadata"]["action"]] == source
    assert sample["events"] == [
        {"kind": "rewind", "start": 4, "end": 8, "start_time": 0.2, "end_time": 0.4}
    ]
    assert len(base.calls) == 4  # repeated frames decode only once in this request
    assert not ds.get_frame(2, 2)["events"]  # no future event metadata


def test_freeze_repeats_state_and_does_not_cross_episodes(base):
    ds = ProgressDataset(
        manual_config(augmentation=[{"name": "freeze", "min_frames": 3, "max_frames": 3}], seed=7),
        base=base,
    )
    sample = ds.get_history(9, ds.episode_length(9) - 1)
    assert len(sample["frame_indices"]) == 6
    freeze = [i for i, kind in enumerate(sample["augmentation_kind"]) if kind == "freeze"]
    assert len(freeze) == 3
    anchor = freeze[0] - 1
    for position in freeze:
        assert torch.equal(sample["images"][CAMERA][anchor], sample["images"][CAMERA][position])
        assert sample["source_timestamps"][anchor] == sample["source_timestamps"][position]
    assert sample["source_indices"].min() >= 4


def test_augmentation_is_repeatable_and_can_be_disabled(base):
    cfg = manual_config(
        augmentation=[
            {"name": "rewind", "min_frames": 2, "max_frames": 3},
            {"name": "freeze", "min_frames": 1, "max_frames": 2},
        ],
        seed=17,
    )
    assert ProgressDataset(cfg, base=base).timelines == ProgressDataset(cfg, base=base).timelines
    disabled = replace(cfg, augmentation=[{"name": "freeze", "enabled": False}])
    assert len(ProgressDataset(disabled, base=base)) == 7
    zero = replace(cfg, augmentation=[{"name": "freeze", "probability": 0}])
    assert len(ProgressDataset(zero, base=base)) == 7


def test_variable_length_batch_masks_and_padding(base):
    ds = ProgressDataset(manual_config(access=AccessConfig("history")), base=base)
    batch = next(iter(make_dataloader(ds, batch_size=3)))
    assert batch["images"][CAMERA].shape == (3, 3, 3, 8, 10)
    assert batch["lengths"].tolist() == [1, 2, 3]
    assert batch["valid_mask"].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    assert batch["source_frame_indices"].tolist() == [[0, -1, -1], [0, 1, -1], [0, 1, 2]]
    assert torch.isnan(batch["timestamps"][0, 1:]).all()
    assert torch.count_nonzero(batch["images"][CAMERA][0, 1:]) == 0


def test_preprocessing_is_configurable_and_has_expected_values(base):
    config = manual_config(
        preprocessing=[
            {"name": "resize", "size": [6, 6]},
            {"name": "center_crop", "size": [4, 4]},
            {"name": "to_float"},
            {"name": "normalize", "mean": [0.1] * 3, "std": [0.2] * 3},
        ]
    )
    sample = ProgressDataset(config, base=base).get_frame(2, 2)
    assert sample["images"][CAMERA].shape == (1, 3, 4, 4)
    torch.testing.assert_close(sample["images"][CAMERA], torch.full((1, 3, 4, 4), 0.5))


def test_extensions_use_the_same_configuration_interface(base):
    @register_preprocessing("test_double")
    class Double:
        def __call__(self, frames):
            return frames * 2

    @register_augmentation("test_identity")
    class Identity:
        def __call__(self, timeline: Timeline, rng):
            return timeline

    ds = ProgressDataset(
        manual_config(
            preprocessing=[{"name": "test_double"}], augmentation=[{"name": "test_identity"}]
        ),
        base=base,
    )
    assert ds.get_frame(2, 1)["images"][CAMERA][0, 0, 0, 0].item() == pytest.approx(0.2)


@pytest.mark.parametrize(
    "options",
    [
        {"cameras": ["missing"]},
        {"metadata_fields": ["missing"]},
        {"preprocessing": [{"name": "missing"}]},
        {"augmentation": [{"name": "freeze", "probability": 2}]},
        {"augmentation": [{"name": "rewind", "min_frames": 0}]},
    ],
)
def test_invalid_config_fails_early(base, options):
    with pytest.raises(ValueError):
        ProgressDataset(manual_config(**options), base=base)


def test_invalid_access_and_indices(base):
    ds = ProgressDataset(manual_config(), base=base)
    for index in (-1, 7):
        with pytest.raises(IndexError):
            ds[index]
    with pytest.raises(IndexError):
        ds.get_frame(2, 4)
    with pytest.raises(ValueError):
        ds.get_history(2, 2, stride=0)


def test_yaml_config():
    config = load_config("configs/prepare_coffee.yaml")
    assert config.access.mode == "frame"
    assert config.preprocessing[0]["name"] == "resize"
    assert len(config.cameras) == 3


def test_dataloader_keeps_three_camera_views_separate():
    from types import SimpleNamespace

    from conftest import FakeLeRobotDataset

    cameras = ["observation.images.left", "observation.images.right", "observation.images.wrist"]

    class ThreeViews(FakeLeRobotDataset):
        meta = SimpleNamespace(camera_keys=cameras)

        def __getitem__(self, index):
            return {camera: torch.full((3, 8, 10), float(i)) for i, camera in enumerate(cameras)}

    ds = ProgressDataset(manual_config(), base=ThreeViews())
    batch = next(iter(make_dataloader(ds, batch_size=2)))
    assert list(batch["images"]) == cameras
    for index, camera in enumerate(cameras):
        assert batch["images"][camera].shape == (2, 1, 3, 8, 10)
        assert torch.all(batch["images"][camera] == index)
