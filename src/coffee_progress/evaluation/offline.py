"""Full-test evaluation at subtask boundaries, plus controlled robustness diagnostics."""

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..config import AccessConfig
from .estimator import EpisodeContext, as_estimator
from .metrics import OfflineConfig, boundary_frames, episode_metrics, sample_stage_pairs, summarize
from .report import export_videos, save_episode, write_report
from .runner import ObservationBuffer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_subtasks(dataset, episode_id: int):
    """Read privileged annotations only for scoring; never put them in Observation."""
    rows = dataset.base.hf_dataset.select_columns(["subtask_idx"])[
        list(dataset.episodes[episode_id].source_indices)
    ]["subtask_idx"]
    stages = np.array([int(value) for value in rows])
    boundary_frames(stages)
    return stages


def perturbed_samples(
    samples,
    *,
    condition: str,
    fps: float,
    freeze_at: int,
    freeze_frames: int,
    rewind_start: int,
    rewind_end: int,
):
    """Insert frames causally; retain source IDs and assign a new observation clock."""
    history = []
    timeline_index = 0

    def retime(sample, kind):
        nonlocal timeline_index
        result = {
            **sample,
            "frame_index": timeline_index,
            "frame_indices": torch.tensor([timeline_index]),
            "timestamps": torch.tensor([timeline_index / fps], dtype=torch.float64),
            "kind": kind,
        }
        timeline_index += 1
        return result

    for source_index, sample in enumerate(samples):
        yield retime(sample, "original")
        if condition == "freeze" and source_index == freeze_at:
            for _ in range(freeze_frames):
                yield retime(sample, "freeze")
        if condition == "rewind":
            if rewind_start <= source_index < rewind_end:
                history.append(sample)
            if source_index == rewind_end - 1:
                for previous in reversed(history):
                    yield retime(previous, "rewind")
                history.clear()


def predict_condition(estimator, dataset, episode_id, condition, access, protocol, config):
    if condition not in ("original", "freeze", "rewind"):
        raise ValueError("Condition must be original, freeze, or rewind")
    seed_everything(config.seed + episode_id)
    estimator.reset(EpisodeContext(episode_id, dataset.fps, dataset.cameras))
    buffer = ObservationBuffer(access)
    samples = dataset.iter_episode_frames(episode_id)
    result = {key: [] for key in ("progress", "source_frame_indices", "timestamps", "kind")}
    for sample in perturbed_samples(
        samples,
        condition=condition,
        fps=dataset.fps,
        freeze_frames=config.freeze_frames,
        **protocol,
    ):
        try:
            value = estimator.predict(buffer.append(sample))
            if isinstance(value, bool):
                raise ValueError("Predictor returned bool")
            score = float(value)
            if not np.isfinite(score):
                raise ValueError("Predictor returned non-finite progress")
        except Exception as exc:
            raise RuntimeError(
                f"Episode {episode_id}, {condition}, frame {sample['frame_index']}"
            ) from exc
        result["progress"].append(score)
        result["source_frame_indices"].append(int(sample["source_frame_indices"][-1]))
        result["timestamps"].append(float(sample["timestamps"][-1]))
        result["kind"].append(sample["kind"])
    return result


def _save_predictions(path, conditions):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "frame_index",
                "source_frame_index",
                "timestamp_seconds",
                "progress",
                "kind",
            ]
        )
        for condition, result in conditions.items():
            writer.writerows(
                (condition, i, source, timestamp, score, kind)
                for i, (source, timestamp, score, kind) in enumerate(
                    zip(
                        result["source_frame_indices"],
                        result["timestamps"],
                        result["progress"],
                        result["kind"],
                        strict=True,
                    )
                )
            )


def save_summary_csv(summary: dict, path: Path):
    """Spreadsheet-friendly aggregate metrics; run instructions belong in the README."""
    fields = [
        "run_name",
        "seed",
        "test_episodes",
        "metric",
        "value",
        "ci95_lower",
        "ci95_upper",
        "n_episodes",
        "status",
        "note",
    ]
    common = {
        "run_name": summary["name"],
        "seed": summary["seed"],
        "test_episodes": summary["test_episodes"],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, result in summary["metrics"].items():
            interval = result["ci95"] or [None, None]
            writer.writerow(
                {
                    **common,
                    "metric": metric,
                    "value": result["value"],
                    "ci95_lower": interval[0],
                    "ci95_upper": interval[1],
                    "n_episodes": result["n_episodes"],
                    "status": "ok" if result["value"] is not None else "undefined",
                }
            )
        for metric, reason in summary.get("unavailable", {}).items():
            writer.writerow(
                {
                    **common,
                    "metric": metric,
                    "n_episodes": 0,
                    "status": "unavailable",
                    "note": reason,
                }
            )


def evaluate_offline(
    estimator,
    dataset,
    output_dir,
    *,
    name="Progress estimator",
    config: OfflineConfig | None = None,
    access: AccessConfig | None = None,
    model_metadata: dict | None = None,
) -> Path:
    """Evaluate all test episodes and also build the five-video qualitative report."""
    config = config or OfflineConfig()
    access = access or dataset.config.access
    if dataset.split != "test" or dataset.split_manifest is None or dataset.augmentations:
        raise ValueError("Offline evaluation requires the named, unaugmented test dataset")
    if set(dataset.cameras) != set(dataset.base.meta.camera_keys):
        raise ValueError("Offline benchmark requires all source camera views")
    expected = dataset.split_manifest["episodes"]["test"]
    if set(dataset.episode_ids) != set(expected):
        raise ValueError("Offline evaluation must include the entire saved test split")
    # Validate the entire boundary protocol before running any expensive inference.
    stages = {i: read_subtasks(dataset, i) for i in dataset.episode_ids}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    offline, qualitative = output / "offline", output / "qualitative"
    predictions = offline / "predictions"
    predictions.mkdir(parents=True)
    qualitative.mkdir()
    adapter = as_estimator(estimator)
    rows, anchors, boundary_rows, qualitative_results, protocols = [], [], [], [], {}
    pair_rows = []
    started = perf_counter()
    with torch.inference_mode():
        for position, episode_id in enumerate(dataset.episode_ids):
            print(f"Offline {position + 1}/{len(expected)}: episode {episode_id}", flush=True)
            boundaries = boundary_frames(stages[episode_id])
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, episode_id]))
            center = int(rng.choice(boundaries))
            length = min(config.rewind_frames, len(stages[episode_id]))
            start = max(0, min(center - length // 2, len(stages[episode_id]) - length))
            protocol = {
                "freeze_at": int(rng.integers(0, len(stages[episode_id]))),
                "rewind_start": start,
                "rewind_end": start + length,
            }
            protocols[episode_id] = protocol
            conditions = {
                condition: predict_condition(
                    adapter, dataset, episode_id, condition, access, protocol, config
                )
                for condition in ("original", "freeze", "rewind")
            }
            original = conditions["original"]
            pairs = sample_stage_pairs(stages[episode_id], config, episode_id)
            metrics, milestones = episode_metrics(
                original["progress"],
                stages[episode_id],
                conditions["freeze"],
                conditions["rewind"],
                config,
                stage_pairs=pairs,
            )
            for earlier, later in pairs:
                pair_rows.append(
                    {
                        "episode_id": episode_id,
                        "earlier_frame": int(earlier),
                        "later_frame": int(later),
                        "earlier_subtask": int(stages[episode_id][earlier]),
                        "later_subtask": int(stages[episode_id][later]),
                        "correct": original["progress"][later]
                        > original["progress"][earlier] + config.tie_epsilon,
                    }
                )
            rows.append(metrics)
            anchors.append(milestones)
            _save_predictions(predictions / f"episode_{episode_id:06d}.csv", conditions)
            for index, (frame, score) in enumerate(zip(boundaries, milestones, strict=True)):
                boundary_rows.append(
                    {
                        "episode_id": episode_id,
                        "milestone": index + 1,
                        "frame_index": int(frame),
                        "timestamp_seconds": frame / dataset.fps,
                        "progress": score,
                    }
                )
            if episode_id in dataset.split_manifest["qualitative_episodes"]:
                result = {
                    "episode_id": episode_id,
                    "fps": dataset.fps,
                    "frame_indices": list(range(len(original["progress"]))),
                    "timestamps": original["timestamps"],
                    "progress": original["progress"],
                    "boundaries": boundaries.tolist(),
                }
                save_episode(result, qualitative, name)
                result["videos"] = export_videos(dataset, episode_id, qualitative)
                qualitative_results.append(result)
    summary = {
        "name": name,
        "seed": config.seed,
        "test_episodes": len(expected),
        "test_frames": sum(len(x) for x in stages.values()),
        "elapsed_seconds": perf_counter() - started,
        "protocol": asdict(config),
        "perturbation_locations": protocols,
        "input_access": asdict(access),
        "dataset_config": asdict(dataset.config),
        "split_manifest": dataset.split_manifest,
        "model_metadata": model_metadata or {},
        "metrics": summarize(rows, anchors, config),
        "metric_protocol_version": 2,
        "prediction_conditions": ["original", "freeze", "rewind"],
    }
    (offline / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for file, records in (
        (
            "metrics.csv",
            [{"episode_id": i, **row} for i, row in zip(dataset.episode_ids, rows, strict=True)],
        ),
        ("boundaries.csv", boundary_rows),
        ("stage_pairs.csv", pair_rows),
    ):
        with (offline / file).open("w", newline="", encoding="utf-8") as handle:
            fields = (
                list(records[0])
                if records
                else [
                    "episode_id",
                    "earlier_frame",
                    "later_frame",
                    "earlier_subtask",
                    "later_subtask",
                    "correct",
                ]
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    save_summary_csv(summary, offline / "summary.csv")
    write_report(
        {
            "schema_version": 1,
            "name": name,
            "signal": "raw predicted progress",
            "input_access": asdict(access),
            "dataset_config": asdict(dataset.config),
            "split_manifest": dataset.split_manifest,
            "model_metadata": model_metadata or {},
            "episodes": qualitative_results,
        },
        qualitative,
    )
    print(f"Offline metrics ready: {offline / 'summary.csv'}", flush=True)
    return offline / "summary.json"
