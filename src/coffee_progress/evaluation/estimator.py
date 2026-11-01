"""Small adapters keep model architectures, checkpoints, and services out of the runner."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class EpisodeContext:
    """Provided once per episode; deliberately excludes outcomes and future frames."""

    episode_id: int
    fps: float
    cameras: tuple[str, ...]


@dataclass(frozen=True)
class Observation:
    """Preprocessed causal input. The last image in each clip is the queried frame.

    images: camera -> float/RGB tensor [T,C,H,W], as configured by the dataset.
    frame_indices/timestamps: [T], in ascending observation order. No actions,
    subtask labels, outcomes, or episode duration are sent to the estimator.
    """

    images: Mapping[str, torch.Tensor]
    frame_indices: torch.Tensor
    timestamps: torch.Tensor


class ProgressEstimator(Protocol):
    """Implement these methods, or use CallableEstimator for an existing predictor.

    predict returns one finite scalar for the last frame, conventionally in [0,1].
    The runner saves raw values without clamping, smoothing, or accumulation.
    """

    def reset(self, context: EpisodeContext) -> None: ...

    def predict(self, observation: Observation) -> float: ...


class CallableEstimator:
    """Adapt a callable/remote request; an optional hook resets state or loads references."""

    def __init__(
        self,
        predict: Callable[[Observation], float],
        reset: Callable[[EpisodeContext], None] | None = None,
    ):
        self._predict = predict
        self._reset = reset

    def reset(self, context: EpisodeContext) -> None:
        if self._reset is not None:
            self._reset(context)

    def predict(self, observation: Observation) -> float:
        return self._predict(observation)


def as_estimator(value) -> CallableEstimator:
    """Accept a protocol implementation, a predict-only object, or a plain callable."""
    predict = getattr(value, "predict", None)
    if predict is None:
        predict = value
    if not callable(predict):
        raise TypeError("Estimator must be callable or provide predict(observation)")
    reset = getattr(value, "reset", None)
    if reset is not None and not callable(reset):
        raise TypeError("Estimator reset must be callable")
    return CallableEstimator(predict, reset)
