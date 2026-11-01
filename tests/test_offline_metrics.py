import numpy as np
import pytest

from coffee_progress.evaluation.metrics import (
    OfflineConfig,
    boundary_frames,
    episode_metrics,
    sample_stage_pairs,
    summarize,
)


def fixtures_for_scores(scores):
    stages = np.repeat(np.arange(4), 4)
    freeze_source = list(range(7)) + [6, 6] + list(range(7, 16))
    rewind_source = list(range(12)) + list(range(4, 12))[::-1] + list(range(12, 16))
    freeze = {
        "progress": np.asarray(scores)[freeze_source].tolist(),
        "kind": ["original"] * 7 + ["freeze"] * 2 + ["original"] * 9,
    }
    rewind = {
        "progress": np.asarray(scores)[rewind_source].tolist(),
        "source_frame_indices": rewind_source,
        "kind": ["original"] * 12 + ["rewind"] * 8 + ["original"] * 4,
    }
    return stages, freeze, rewind


def test_exact_transition_frame_and_ideal_predictor():
    scores = np.repeat(np.arange(4), 4) / 3
    stages, freeze, rewind = fixtures_for_scores(scores)
    assert boundary_frames(stages).tolist() == [4, 8, 12]
    metrics, anchors = episode_metrics(scores, stages, freeze, rewind, OfflineConfig())
    assert anchors == pytest.approx([1 / 3, 2 / 3, 1])
    assert metrics["freeze_drift"] == 0
    assert metrics["stage_ordering_accuracy"] == metrics["rewind_ordering_accuracy"] == 1
    assert metrics["progress_range"] == 1


def test_constant_predictor_cannot_win_ordering():
    scores = np.full(16, 0.5)
    stages, freeze, rewind = fixtures_for_scores(scores)
    metrics, _ = episode_metrics(scores, stages, freeze, rewind, OfflineConfig())
    assert metrics["stage_ordering_accuracy"] == metrics["rewind_ordering_accuracy"] == 0
    assert metrics["progress_range"] == 0


def test_bootstrap_reproducible_at_episode_level():
    rows = [{"ordering": 0.0}, {"ordering": 1.0}, {"ordering": 0.5}]
    anchors = [[0.0, 0.5, 1.0], [0.1, 0.6, 1.0], [0.2, 0.7, 0.8]]
    cfg = OfflineConfig(bootstrap_samples=100)
    a = summarize(rows, anchors, cfg)
    assert a == summarize(rows, anchors, cfg)
    assert a["ordering"]["value"] == 0.5
    assert a["ordering"]["n_episodes"] == 3
    assert a["cross_episode_milestone_std"]["value"] > 0


def test_pair_sampling_is_unique_reproducible_and_excludes_same_subtask():
    stages = np.repeat(np.arange(4), 4)
    config = OfflineConfig(stage_pairs=20)
    pairs = sample_stage_pairs(stages, config, episode_id=51)
    assert np.array_equal(pairs, sample_stage_pairs(stages, config, episode_id=51))
    assert len(set(map(tuple, pairs))) == len(pairs)
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert np.all(stages[pairs[:, 0]] != stages[pairs[:, 1]])


def test_sampling_all_pairs_does_not_balance_stages():
    stages = np.array([0] * 8 + [1] * 2 + [2] + [3])
    pairs = sample_stage_pairs(stages, OfflineConfig())
    expected = {
        (i, j)
        for i in range(len(stages))
        for j in range(i + 1, len(stages))
        if stages[i] != stages[j]
    }
    assert set(map(tuple, pairs)) == expected
    assert sum(stages[i] == 0 and stages[j] == 1 for i, j in pairs) == 16
    assert sum(stages[i] == 2 and stages[j] == 3 for i, j in pairs) == 1


def test_no_eligible_sampled_pairs_is_undefined_not_zero():
    scores = np.full(16, 0.5)
    stages, freeze, rewind = fixtures_for_scores(scores)
    metrics, _ = episode_metrics(
        scores, stages, freeze, rewind, OfflineConfig(), stage_pairs=np.empty((0, 2), dtype=int)
    )
    assert metrics["stage_ordering_accuracy"] is None


def test_unexpected_boundary_protocol_is_rejected():
    with pytest.raises(ValueError, match="exactly"):
        boundary_frames([0, 0, 1, 2, 2])
    with pytest.raises(ValueError, match="nonnegative"):
        OfflineConfig(seed=-1)
