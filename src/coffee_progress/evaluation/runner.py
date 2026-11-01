"""Score every frame of the five fixed held-out episodes, in causal order."""

import math
from collections import deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from ..config import AccessConfig
from ..dataset import ProgressDataset
from .estimator import EpisodeContext, Observation, as_estimator
from .report import export_videos, save_episode, write_report


class ObservationBuffer:
    """Decode each frame once, even when the estimator requests overlapping histories."""

    def __init__(self, access: AccessConfig):
        self.access = access
        capacity = (
            1
            if access.mode == "frame"
            else (
                None if access.max_frames is None else (access.max_frames - 1) * access.stride + 1
            )
        )
        self.frames = deque(maxlen=capacity)

    def append(self, sample: dict) -> Observation:
        self.frames.append(sample)
        selected = list(self.frames)[:: -self.access.stride][::-1]
        return Observation(
            images={
                camera: torch.cat([s["images"][camera] for s in selected], dim=0)
                for camera in sample["images"]
            },
            frame_indices=torch.cat([s["frame_indices"] for s in selected]),
            timestamps=torch.cat([s["timestamps"] for s in selected]),
        )


def evaluate_qualitative(
    estimator,
    dataset: ProgressDataset,
    output_dir: str | Path,
    *,
    name: str = "Progress estimator",
    access: AccessConfig | None = None,
    model_metadata: dict | None = None,
) -> Path:
    """Write CSV, PNG, run metadata, and a portable HTML/video report.

    Always uses the saved five qualitative test episodes, with augmentation off.
    A fresh output directory prevents accidental replacement of previous results.
    Estimators run in the main process under torch.inference_mode; their adapters
    own model.eval(), device placement, input/output conversion, and API handling.
    """
    if dataset.split != "test" or dataset.split_manifest is None or dataset.augmentations:
        raise ValueError("Qualitative evaluation requires a named, unaugmented test dataset")
    episode_ids = dataset.split_manifest["qualitative_episodes"]
    if len(episode_ids) != 5 or not set(episode_ids) <= set(dataset.episode_ids):
        raise ValueError("Dataset must include the five fixed qualitative test episodes")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    access = access or dataset.config.access
    adapter = as_estimator(estimator)
    results = []
    with torch.inference_mode():
        for episode_id in episode_ids:
            print(f"Evaluating episode {episode_id}...", flush=True)
            adapter.reset(EpisodeContext(episode_id, dataset.fps, dataset.cameras))
            buffer = ObservationBuffer(access)
            scores, timestamps = [], []
            started = perf_counter()
            for frame_index, sample in enumerate(dataset.iter_episode_frames(episode_id)):
                observation = buffer.append(sample)
                try:
                    value = adapter.predict(observation)
                    if isinstance(value, torch.Tensor) and value.numel() != 1:
                        raise ValueError("Expected one scalar progress value")
                    if isinstance(value, bool):
                        raise ValueError("Expected continuous progress, received bool")
                    score = float(value)
                    if not math.isfinite(score):
                        raise ValueError("Progress must be finite")
                except Exception as exc:
                    raise RuntimeError(
                        f"Estimator failed at episode {episode_id}, frame {frame_index}"
                    ) from exc
                scores.append(score)
                timestamps.append(float(sample["source_timestamps"][-1]))
            result = {
                "episode_id": episode_id,
                "fps": dataset.fps,
                "frame_indices": list(range(len(scores))),
                "timestamps": timestamps,
                "progress": scores,
                "elapsed_seconds": perf_counter() - started,
            }
            save_episode(result, output, name)
            result["videos"] = export_videos(dataset, episode_id, output)
            results.append(result)
    run = {
        "schema_version": 1,
        "name": name,
        "signal": "raw predicted progress",
        "input_access": asdict(access),
        "dataset_config": asdict(dataset.config),
        "split_manifest": dataset.split_manifest,
        "model_metadata": model_metadata or {},
        "episodes": results,
    }
    return write_report(run, output)
