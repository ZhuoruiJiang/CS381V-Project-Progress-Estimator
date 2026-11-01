"""Static plots and a local, dependency-free interactive video report."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_episode(result: dict, output: Path, name: str):
    stem = f"episode_{result['episode_id']:06d}"
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds", "progress"])
        writer.writerows(
            zip(result["frame_indices"], result["timestamps"], result["progress"], strict=True)
        )
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(result["frame_indices"], result["progress"], color="#177d94", linewidth=2)
    ax.set(xlabel="Timestep", ylabel="Predicted progress")
    ax.set_title(f"Episode {result['episode_id']}  |  {name}", loc="left", pad=16)
    values = result["progress"]
    low, high = min(0, min(values)), max(1, max(values))
    ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / f"{stem}.png", dpi=180)
    plt.close(fig)


def export_videos(dataset, episode_id: int, output: Path) -> dict[str, str]:
    """Copy original v2.1 MP4s so the report survives moving/sharing its whole folder."""
    assets = output / "videos"
    assets.mkdir(exist_ok=True)
    paths = {}
    for camera_index, camera in enumerate(dataset.cameras):
        source = dataset.base.root / dataset.base.meta.get_video_file_path(episode_id, camera)
        target = assets / f"episode_{episode_id:06d}_camera_{camera_index}.mp4"
        shutil.copyfile(source, target)
        paths[camera] = target.relative_to(output).as_posix()
    return paths


def write_report(run: dict, output: Path) -> Path:
    payload = json.dumps(run, indent=2, allow_nan=False)
    (output / "predictions.json").write_text(payload + "\n", encoding="utf-8")
    assets = Path(__file__).parent / "assets"
    template = (assets / "viewer.html").read_text(encoding="utf-8")
    # Escape HTML-sensitive characters so model names/metadata cannot break out of JSON.
    embedded = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.replace("__STYLES__", (assets / "viewer.css").read_text(encoding="utf-8"))
    html = html.replace("__SCRIPT__", (assets / "viewer.js").read_text(encoding="utf-8"))
    html = html.replace("__REPORT_DATA__", embedded)
    index = output / "index.html"
    index.write_text(html, encoding="utf-8")
    print(f"Report ready: {index}", flush=True)
    return index
