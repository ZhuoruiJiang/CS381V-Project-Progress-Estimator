"""Run a user-provided estimator factory on the fixed qualitative test episodes."""

import argparse
import importlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath
from zoneinfo import ZoneInfo

import yaml

from ..config import AccessConfig, load_config
from ..dataset import ProgressDataset
from .metrics import OfflineConfig
from .offline import evaluate_offline, seed_everything
from .runner import evaluate_qualitative


def output_directory(mode: str, output: str | None = None, *, now: datetime | None = None) -> Path:
    """Append a Chicago-local timestamp to an optional relative run name/path."""
    if mode not in ("qualitative", "offline"):
        raise ValueError("mode must be qualitative or offline")
    timestamp = (now or datetime.now(ZoneInfo("America/Chicago"))).strftime("%Y-%m-%d-%H-%M-%S")
    root = Path("outputs/evaluations")
    if not output:
        return root / timestamp
    relative = Path(output.replace("\\", "/"))
    if (
        relative.anchor
        or PureWindowsPath(output).drive
        or not relative.parts
        or ".." in relative.parts
        or any(character in '<>:"|?*' for character in output)
    ):
        raise ValueError(
            "--output must be a relative run name/path without '..' or reserved characters"
        )
    return root / relative.parent / f"{relative.name}_{timestamp}"


def load_factory(spec: str):
    """Accept package.module:factory or path/to/file.py:factory."""
    module_name, separator, attribute = spec.rpartition(":")
    if not separator or not attribute:
        raise ValueError("Use module:factory or path/to/file.py:factory")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        module_spec = importlib.util.spec_from_file_location("coffee_user_estimator", path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"{attribute} must be a callable factory")
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", required=True, help="module:factory or file.py:factory")
    parser.add_argument("--config", default="configs/prepare_coffee.yaml")
    parser.add_argument("--output", help="Optional run name/path under outputs/evaluations")
    parser.add_argument("--name", default="Progress estimator")
    parser.add_argument("--kwargs", default="{}", help="JSON keyword arguments passed to factory")
    parser.add_argument("--input-mode", choices=("frame", "history"))
    parser.add_argument("--history-stride", type=int)
    parser.add_argument("--history-max-frames", type=int)
    parser.add_argument(
        "--mode",
        choices=("qualitative", "offline"),
        default="offline",
        help="Evaluation scope (default: offline, all test episodes)",
    )
    parser.add_argument("--protocol", default="configs/offline_evaluation.yaml")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    options = json.loads(args.kwargs)
    if not isinstance(options, dict):
        parser.error("--kwargs must be a JSON object")
    access = AccessConfig(
        mode=args.input_mode or config.access.mode,
        stride=args.history_stride if args.history_stride is not None else config.access.stride,
        max_frames=(
            args.history_max_frames
            if args.history_max_frames is not None
            else config.access.max_frames
        ),
    )
    try:
        output = output_directory(args.mode, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    if output.exists():
        parser.error(f"Run directory already exists: {output}. Choose another name or retry later.")
    print(f"Run folder: {output}", flush=True)
    import torch

    torch.set_num_threads(args.torch_threads)
    offline_config = OfflineConfig(
        **yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    )
    seed_everything(offline_config.seed)
    estimator = load_factory(args.estimator)(**options)
    dataset = ProgressDataset(config, split="test")
    evaluator = evaluate_offline if args.mode == "offline" else evaluate_qualitative
    extra = {"config": offline_config} if args.mode == "offline" else {}
    evaluator(
        estimator,
        dataset,
        output if args.mode == "offline" else output / "qualitative",
        name=args.name,
        access=access,
        model_metadata={"factory": args.estimator, "factory_kwargs": options},
        **extra,
    )
    print(f"Open viewer: uv run coffee-view {output}", flush=True)


if __name__ == "__main__":
    main()
