"""Five offline metrics using original, freeze, and rewind predictions."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class OfflineConfig:
    seed: int = 0
    bootstrap_samples: int = 1000
    freeze_frames: int = 10
    rewind_frames: int = 16
    stage_pairs: int = 1000
    tie_epsilon: float = 1e-6

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.bootstrap_samples < 1 or self.freeze_frames < 1 or self.rewind_frames < 2:
            raise ValueError("Bootstrap/freeze counts must be positive; rewind needs >=2 frames")
        if self.stage_pairs < 1:
            raise ValueError("stage_pairs must be positive")
        if not np.isfinite(self.tie_epsilon) or self.tie_epsilon < 0:
            raise ValueError("tie_epsilon must be finite and nonnegative")


def boundary_frames(stages) -> np.ndarray:
    """Return the first frame of each new subtask segment, including done."""
    stages = np.asarray(stages, dtype=int)
    if stages.ndim != 1 or len(stages) < 2:
        raise ValueError("Need a one-dimensional per-frame subtask sequence")
    changes = np.flatnonzero(stages[1:] != stages[:-1]) + 1
    if len(changes) != 3 or stages[0] != 0 or not np.array_equal(stages[changes], [1, 2, 3]):
        raise ValueError("PrepareCoffee protocol requires exactly 0 -> 1 -> 2 -> 3(done)")
    return changes


def sample_stage_pairs(stages, config: OfflineConfig, episode_id: int = 0) -> np.ndarray:
    """Uniformly draw distinct unordered frame pairs, then exclude same-subtask pairs.

    Sampling is independent of predictions and perturbation placement. Stage pairs
    are not balanced or resampled. Triangular indexing avoids materializing N*N pairs.
    """
    stages = np.asarray(stages)
    count = len(stages)
    total = count * (count - 1) // 2
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, episode_id, 1]))
    draws = rng.choice(total, size=min(config.stage_pairs, total), replace=False)
    indices = np.arange(count)
    row_starts = indices * (2 * count - indices - 1) // 2
    earlier = np.searchsorted(row_starts, draws, side="right") - 1
    later = earlier + 1 + draws - row_starts[earlier]
    pairs = np.column_stack([earlier, later])
    return pairs[stages[earlier] != stages[later]]


def episode_metrics(baseline, stages, freeze, rewind, config: OfflineConfig, *, stage_pairs=None):
    baseline = np.asarray(baseline, dtype=float)
    boundaries = boundary_frames(stages)
    anchors = baseline[boundaries]
    pairs = sample_stage_pairs(stages, config) if stage_pairs is None else stage_pairs
    inserted = np.flatnonzero(np.asarray(rewind["kind"]) == "rewind")
    source = np.asarray(rewind["source_frame_indices"])[inserted]
    rewound_scores = np.asarray(rewind["progress"])[inserted]
    cross_stage = [
        (i, j)
        for i, j in combinations(range(len(source)), 2)
        if stages[source[i]] > stages[source[j]]
    ]
    frozen = np.flatnonzero(np.asarray(freeze["kind"]) == "freeze")
    delta = freeze["progress"][frozen[-1]] - freeze["progress"][frozen[0] - 1]
    return {
        "stage_ordering_accuracy": float(
            np.mean(baseline[pairs[:, 1]] > baseline[pairs[:, 0]] + config.tie_epsilon)
        )
        if len(pairs)
        else None,
        "freeze_drift": abs(float(delta)),
        "rewind_ordering_accuracy": float(
            np.mean(
                [rewound_scores[i] > rewound_scores[j] + config.tie_epsilon for i, j in cross_stage]
            )
        )
        if cross_stage
        else None,
        "progress_range": float(np.ptp(baseline)),
    }, anchors.tolist()


def summarize(rows: list[dict], anchors: list[list[float]], config: OfflineConfig) -> dict:
    """Bootstrap whole episodes, preserving every within-episode dependency."""
    rng = np.random.default_rng(config.seed)
    summary = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        if not len(values):
            summary[key] = {"value": None, "ci95": None, "n_episodes": 0}
            continue
        draws = rng.integers(0, len(values), size=(config.bootstrap_samples, len(values)))
        intervals = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
        summary[key] = {
            "value": float(values.mean()),
            "ci95": intervals.tolist(),
            "n_episodes": len(values),
        }
    values = np.asarray(anchors)
    draws = rng.integers(0, len(values), size=(config.bootstrap_samples, len(values)))
    std_samples = values[draws].std(axis=1).mean(axis=1)
    summary["cross_episode_milestone_std"] = {
        "value": float(values.std(axis=0).mean()),
        "ci95": np.quantile(std_samples, [0.025, 0.975]).tolist(),
        "n_episodes": len(values),
        "per_milestone_mean": values.mean(axis=0).tolist(),
        "per_milestone_std": values.std(axis=0).tolist(),
    }
    return summary
