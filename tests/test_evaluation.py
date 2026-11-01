import csv
import json
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
import torch
from conftest import CAMERA

from coffee_progress import AccessConfig, DatasetConfig, ProgressDataset, SourceConfig
from coffee_progress.evaluation import CallableEstimator, evaluate_qualitative
from coffee_progress.evaluation.cli import load_factory
from coffee_progress.evaluation.estimator import EpisodeContext, Observation
from coffee_progress.evaluation.report import write_report


@pytest.fixture
def evaluation_dataset(base, tmp_path):
    ids = [2, 9, 17, 23, 31]
    base.hf_dataset = datasets.Dataset.from_dict(
        {
            "episode_index": [i for i in ids for _ in range(4)],
            "frame_index": list(range(4)) * 5,
            "timestamp": [0.0, 0.05, 0.1, 0.15] * 5,
        }
    )
    base.root = tmp_path
    base.meta = SimpleNamespace(
        camera_keys=[CAMERA], get_video_file_path=lambda episode, camera: Path(f"{episode}.mp4")
    )
    for episode in ids:
        (tmp_path / f"{episode}.mp4").write_bytes(b"video-copy-fixture")
    ds = ProgressDataset(DatasetConfig(source=SourceConfig(split=None)), base=base)
    # Exercise the runner independently of split generation (covered in test_splits).
    ds.split = "test"
    ds.split_manifest = {"qualitative_episodes": ids, "episodes": {"test": ids}}
    return ds


def test_every_frame_scored_in_order_with_causal_history_and_reset(evaluation_dataset, tmp_path):
    resets, observations = [], []

    def predict(observation):
        assert isinstance(observation, Observation)
        assert not hasattr(observation, "metadata")
        assert not hasattr(observation, "episode_length")
        observations.append(observation.frame_indices.tolist())
        return torch.tensor(float(observation.frame_indices[-1]) / 3)

    estimator = CallableEstimator(predict, reset=lambda context: resets.append(context))
    output = tmp_path / "report"
    result = evaluate_qualitative(
        estimator,
        evaluation_dataset,
        output,
        name="Test model",
        access=AccessConfig("history", stride=2, max_frames=2),
    )
    assert result == output / "index.html"
    assert all(isinstance(c, EpisodeContext) for c in resets)
    assert [c.episode_id for c in resets] == list(evaluation_dataset.episode_ids)
    assert observations == [[0], [1], [0, 2], [1, 3]] * 5
    assert len(evaluation_dataset.base.calls) == 20  # history does not re-decode old frames
    run = json.loads((output / "predictions.json").read_text())
    for episode in run["episodes"]:
        assert episode["frame_indices"] == [0, 1, 2, 3]
        assert episode["progress"] == pytest.approx([0, 1 / 3, 2 / 3, 1])
        path = output / f"episode_{episode['episode_id']:06d}"
        with path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4 and float(rows[-1]["progress"]) == 1
        assert path.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
        assert (output / episode["videos"][CAMERA]).read_bytes() == b"video-copy-fixture"


def test_plain_callable_and_raw_scores_are_preserved(evaluation_dataset, tmp_path):
    evaluate_qualitative(lambda obs: -2.5, evaluation_dataset, tmp_path / "raw")
    run = json.loads((tmp_path / "raw/predictions.json").read_text())
    assert run["episodes"][0]["progress"] == [-2.5] * 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, torch.zeros(2)])
def test_invalid_predictions_fail_with_episode_and_frame(evaluation_dataset, tmp_path, value):
    with pytest.raises(RuntimeError, match="episode 2, frame 0"):
        evaluate_qualitative(lambda obs: value, evaluation_dataset, tmp_path / "invalid")
    assert not (tmp_path / "invalid/index.html").exists()


def test_existing_results_are_not_overwritten(evaluation_dataset, tmp_path):
    with pytest.raises(FileExistsError):
        evaluate_qualitative(lambda obs: 0.5, evaluation_dataset, tmp_path)


def test_evaluation_requires_the_fixed_unaugmented_test_set(evaluation_dataset, tmp_path):
    evaluation_dataset.split = "train"
    with pytest.raises(ValueError, match="test dataset"):
        evaluate_qualitative(lambda obs: 0.5, evaluation_dataset, tmp_path / "train")
    evaluation_dataset.split = "test"
    evaluation_dataset.split_manifest["qualitative_episodes"] = [2]
    with pytest.raises(ValueError, match="five fixed"):
        evaluate_qualitative(lambda obs: 0.5, evaluation_dataset, tmp_path / "few")


def test_report_embedded_metadata_cannot_escape_json_script(tmp_path):
    run = {"name": "</script><script>bad()</script>", "episodes": []}
    write_report(run, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<script>bad()" not in html
    assert "\\u003c/script\\u003e" in html


def test_dummy_factory_loads_and_uses_pixels():
    estimator = load_factory("examples/dummy_reward.py:create_estimator")()
    estimator.reset(EpisodeContext(0, 20, (CAMERA,)))
    observation = Observation(
        {CAMERA: torch.ones(1, 3, 8, 8)}, torch.tensor([0]), torch.tensor([0.0])
    )
    assert estimator.predict(observation) == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    with pytest.raises(ValueError, match="module:factory"):
        load_factory("no-factory")


def test_offline_full_split_outputs_are_reproducible(evaluation_dataset, tmp_path, monkeypatch):
    from coffee_progress.evaluation.metrics import OfflineConfig
    from coffee_progress.evaluation.offline import evaluate_offline

    ds = evaluation_dataset
    ds.base.hf_dataset = ds.base.hf_dataset.add_column("subtask_idx", [0, 1, 2, 3] * 5)

    def frames(episode_id, *, image_transform=None):
        for index in range(4):
            sample = ds.get_frame(episode_id, index)
            if image_transform:
                sample["images"] = {c: image_transform(f) for c, f in sample["images"].items()}
            yield sample

    monkeypatch.setattr(ds, "iter_episode_frames", frames)
    config = OfflineConfig(bootstrap_samples=20, freeze_frames=2, rewind_frames=4)

    def estimator(obs):
        return float(next(iter(obs.images.values())).mean())

    a = evaluate_offline(estimator, ds, tmp_path / "run_a", config=config)
    b = evaluate_offline(estimator, ds, tmp_path / "run_b", config=config)
    first, second = json.loads(a.read_text()), json.loads(b.read_text())
    assert first["metrics"] == second["metrics"]
    assert first["perturbation_locations"] == second["perturbation_locations"]
    assert first["test_episodes"] == 5 and first["test_frames"] == 20
    with (tmp_path / "run_a/offline/summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    for row in summary_rows:
        if row["status"] == "ok":
            expected = first["metrics"][row["metric"]]
            assert float(row["value"]) == expected["value"]
            assert float(row["ci95_lower"]) == expected["ci95"][0]
            assert float(row["ci95_upper"]) == expected["ci95"][1]
        assert row["test_episodes"] == "5"
    assert len(summary_rows) == 5
    assert set(first["metrics"]) == {
        "stage_ordering_accuracy",
        "cross_episode_milestone_std",
        "progress_range",
        "freeze_drift",
        "rewind_ordering_accuracy",
    }
    with (tmp_path / "run_a/offline/stage_pairs.csv").open(newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert len(pair_rows) == 30  # all six distinct pairs in each four-frame test episode
    with (tmp_path / "run_a/offline/predictions/episode_000002.csv").open(newline="") as handle:
        conditions = {row["condition"] for row in csv.DictReader(handle)}
    assert conditions == {"original", "freeze", "rewind"}
    assert not (tmp_path / "run_a/offline/report.md").exists()
    assert (tmp_path / "run_a/qualitative/index.html").exists()
    assert len(list((tmp_path / "run_a/offline/predictions").glob("*.csv"))) == 5


def test_offline_rejects_omitted_cameras(evaluation_dataset, tmp_path):
    from coffee_progress.evaluation.offline import evaluate_offline

    evaluation_dataset.base.meta.camera_keys.append("observation.images.omitted")
    with pytest.raises(ValueError, match="all source camera"):
        evaluate_offline(lambda obs: 0.5, evaluation_dataset, tmp_path / "partial")
