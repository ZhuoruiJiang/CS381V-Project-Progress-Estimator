"""Export the five qualitative test videos with their exact per-frame subtask labels."""

import argparse
import csv
import json
import textwrap
from fractions import Fraction
from pathlib import Path

import av
import pyarrow.parquet as pq
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from coffee_progress import ProgressDataset, load_config

LABEL_FIELDS = (
    "subtask_idx",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
)


def annotate_episode(dataset, episode_id: int, camera: str, output: Path, tasks: dict) -> dict:
    meta = dataset.base.meta
    table = pq.read_table(dataset.base.root / meta.get_data_file_path(episode_id)).to_pydict()
    count = len(table["frame_index"])
    if table["frame_index"] != list(range(count)):
        raise ValueError(f"Episode {episode_id}: source frame indices are not contiguous")
    source = dataset.base.root / meta.get_video_file_path(episode_id, camera)
    target = output / f"episode_{episode_id:06d}_subtasks.mp4"
    partial = target.with_suffix(".partial.mp4")
    rate = Fraction(str(dataset.fps))
    font_path = font_manager.findfont("DejaVu Sans")
    fonts = {size: ImageFont.truetype(font_path, size) for size in (18, 20, 24)}
    segments = []
    previous = None
    decoded = 0
    with av.open(str(source)) as reader, av.open(str(partial), "w") as writer:
        stream = writer.add_stream("libx264", rate=rate)
        stream.width, stream.height = 640, 800
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        for index, frame in enumerate(reader.decode(video=0)):
            if index >= count:
                raise ValueError(f"Episode {episode_id}: video has more frames than annotations")
            timestamp = float(table["timestamp"][index])
            if frame.time is None or abs(float(frame.time) - timestamp) > 0.5 / dataset.fps:
                raise ValueError(
                    f"Episode {episode_id}, frame {index}: video/annotation time mismatch"
                )
            subtask = int(table["subtask_idx"][index])
            instruction, skill, stage = (
                tasks[int(table[field][index])] for field in LABEL_FIELDS[1:]
            )
            identity = (subtask, instruction, skill, stage)
            if identity != previous:
                if segments:
                    segments[-1]["end_frame_exclusive"] = index
                segments.append(
                    {
                        "start_frame": index,
                        "subtask_idx": subtask,
                        "instruction": instruction,
                        "skill": skill,
                        "stage": stage,
                    }
                )
                previous = identity
            canvas = Image.new("RGB", (640, 800), "#132b39")
            canvas.paste(frame.to_image().resize((640, 640), Image.Resampling.BILINEAR), (0, 160))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (18, 10),
                f"Episode {episode_id}   |   Frame {index}/{count - 1}   |   {timestamp:.2f}s",
                font=fonts[18],
                fill="#bcd4df",
            )
            draw.text(
                (18, 39),
                f"Subtask index {subtask}   |   Stage: {stage}",
                font=fonts[20],
                fill="#83d9db",
            )
            for line_number, line in enumerate(textwrap.wrap(instruction, width=43)):
                draw.text((18, 70 + line_number * 29), line, font=fonts[24], fill="white")
            draw.text((18, 135), f"Skill: {skill}", font=fonts[18], fill="#bcd4df")
            encoded = av.VideoFrame.from_image(canvas)
            encoded.pts = index
            encoded.time_base = 1 / rate
            for packet in stream.encode(encoded):
                writer.mux(packet)
            decoded += 1
        for packet in stream.encode():
            writer.mux(packet)
    if decoded != count:
        raise ValueError(f"Episode {episode_id}: {decoded} video frames but {count} annotations")
    segments[-1]["end_frame_exclusive"] = count
    partial.rename(target)
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "start_frame",
            "end_frame_exclusive",
            "subtask_idx",
            "instruction",
            "skill",
            "stage",
        )
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(segments)
    print(f"Saved {target}: {count} frames, {len(segments)} segments", flush=True)
    return {"episode_id": episode_id, "frames": count, "video": target.name, "segments": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prepare_coffee.yaml")
    parser.add_argument("--output", default="outputs/annotations/subtasks")
    parser.add_argument("--camera", help="Defaults to the first configured camera")
    args = parser.parse_args()
    dataset = ProgressDataset(load_config(args.config), split="test")
    camera = args.camera or dataset.cameras[0]
    if camera not in dataset.base.meta.camera_keys:
        parser.error(f"Unknown camera: {camera}")
    tasks = {
        row["task_index"]: row["task"]
        for line in (dataset.base.root / "meta/tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and (row := json.loads(line))
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = [
        annotate_episode(dataset, i, camera, output, tasks)
        for i in dataset.split_manifest["qualitative_episodes"]
    ]
    (output / "annotations.json").write_text(
        json.dumps({"camera": camera, "episodes": results}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
