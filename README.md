# PrepareCoffee progress infrastructure

RoboCasa video datasets, configurable transforms, and progress-estimator evaluation.

## Team quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run here:

```sh
uv sync --extra dev
uv run coffee-download
uv run coffee-inspect
uv run coffee-evaluate --mode qualitative --estimator examples/dummy_reward.py:create_estimator --output team_check
```

Open the printed run folder with `uv run coffee-view <run-folder>` in Chrome/Edge.
Then adapt `examples/dummy_reward.py` for your method; the evaluation commands are below.

Python 3.12 is selected automatically. No simulator or GPU is needed. Allow ~2 GB
for the official archive and extracted data, plus the Python environment.

## Load data

```python
from coffee_progress import ProgressDataset, load_config, make_dataloader

config = load_config("configs/prepare_coffee.yaml")
train = ProgressDataset(config)
val = ProgressDataset(config, split="val")
test = ProgressDataset(config, split="test")

episode = train.episode_ids[0]
frame = train.get_frame(episode, 100)
history = train.get_history(episode, 100, stride=4, max_frames=16)
batch = next(iter(make_dataloader(train, batch_size=8, shuffle=True)))
```

- All **three cameras** remain separate: `images[camera]` is `[T,C,H,W]` per sample
  and `[B,T,C,H,W]` per batch. Use `valid_mask` to ignore padded history.
- `dataset[i]` uses the configured frame/history mode. `get_history(..., stride=1,
  max_frames=None)` returns the entire prefix through the requested frame.
- With augmentation, indices refer to its virtual timeline; original source IDs are retained.
- Saved splits: **411 train / 51 val / 52 test**. Reuse `configs/splits/prepare_coffee.json`.
  Validation/test disable augmentation automatically; splitting is by episode, not kitchen.
- For multiple workers, use a `__main__` guard and a fresh dataset. Start with `num_workers=0`.

## Evaluate

Choose an evaluation scope:

- **`qualitative`**: evaluate only the five fixed held-out videos for a faster visual check.
- **`offline`** (default): evaluate all 52 test episodes and the freeze/rewind
  conditions, compute metrics, and produce the same five-video viewer. The viewer reuses
  original predictions from this run; it does not invoke the estimator again.

Quick visual check with the dummy estimator:

```sh
uv run coffee-evaluate --mode qualitative --estimator examples/dummy_reward.py:create_estimator --output my_visual_run
```

Full evaluation on saved videos, seed 0:

```sh
uv run coffee-evaluate --mode offline --estimator examples/dummy_reward.py:create_estimator --output my_metrics_run
```

- Both commands create `outputs/evaluations/<name>_yyyy-mm-dd-hh-mm-ss/`.
  `--output` is optional: omit it for a timestamp-only folder. Times use America/Chicago.
- After either evaluation finishes, run `uv run coffee-view <run-folder>` using the
  printed run folder. It automatically opens the saved viewer in `qualitative/`.
- Videos are copied, not re-encoded. Most runtime comes from evaluating the estimator;
  `qualitative` is faster because it evaluates fewer videos and skips the robustness tests.
- Open the viewer URL in **Chrome/Edge**; keep the terminal running. Drag the slider,
  hover the plot, or play all three synchronized views. Use a new output path per run.
- Offline results: `offline/summary.csv`, per-episode metrics, boundary scores, and all raw
  condition predictions. [Metric definitions](docs/OFFLINE_METRICS.md).
- To plug in your predictor, adapt [examples/dummy_reward.py](examples/dummy_reward.py):
  `predict(obs) -> scalar`, optional `reset(context)`. No particular model library is required.
- Clips/history: `--input-mode history --history-stride 4 --history-max-frames 16`.
  Adapter details and Python API: [docs/EVALUATION.md](docs/EVALUATION.md).
- The dummy averages brightness-based scores across all views; it is only a pipeline check.

### Offline metrics

Five metrics share predictions from **three versions** of each episode: original, freeze,
and rewind. Subtask labels are used only for scoring, never as model inputs.

| Metric | What it measures | Goal |
|---|---|---|
| Stage ordering accuracy | Later subtask scores higher in randomly sampled distinct frame pairs; same-subtask pairs excluded, ties fail | Higher |
| Cross-episode milestone std | Prediction spread across episodes at matching boundaries | Lower |
| Freeze drift | Absolute score change after 10 repeated copies of one frame | Lower |
| Rewind ordering accuracy | In the reversed clip, frames from later source subtasks still score higher than earlier subtasks; ties fail | Higher |
| Progress range | Maximum minus minimum score; zero flags a constant predictor | Diagnostic |

Metrics average episodes equally, with 95% intervals from 1,000 episode-bootstrap samples.
Seed 0 fixes perturbations and resampling. A constant predictor can pass stability tests;
read them alongside ordering and range. Stage ordering draws 1,000 unique candidate pairs
per episode without balancing stages, then excludes same-subtask pairs; scored pairs are
saved in `offline/stage_pairs.csv`. Milestones are the exact first new-label frames, including
task complete; there are no numeric progress targets. [Full protocol](docs/OFFLINE_METRICS.md).

## Configure and extend

Edit [configs/prepare_coffee.yaml](configs/prepare_coffee.yaml): camera selection, frame/history
access, ordered preprocessing, rewind/freeze, optional metadata, and seed.
`[]` disables a transform pipeline. Offline metrics use [configs/offline_evaluation.yaml](configs/offline_evaluation.yaml).

Register a preprocessing callable `[T,C,H,W] -> [T,C,H,W]`, or a temporal augmentation
`(Timeline, rng) -> Timeline`; import it before dataset creation. See
[examples/custom_transforms.py](examples/custom_transforms.py). Original videos are unchanged;
actions/labels stay separate from visual model inputs. No progress training targets are generated.

To export source subtask labels over the five videos: `uv run python examples/annotate_subtasks.py`.

## Outputs and checks

- `outputs/evaluations/<run>_<timestamp>/qualitative/`: viewer, plots, and predictions.
- `outputs/evaluations/<run>_<timestamp>/offline/`: full-test metric CSVs and metadata.
- `outputs/annotations/subtasks/`: labeled videos and segment CSVs.
- `outputs/diagnostics/`: visual checks. `outputs/archive/`: preserved older runs.
- Data, generated outputs, environments, and local credentials are excluded from Git.
  Run the commands above to generate results locally.
- Implemented design: [System Specifications](docs/SYSTEM_SPECIFICATIONS.md).

```sh
uv run pytest -q
uv run ruff check src tests examples
```

Set `COFFEE_INTEGRATION=1` to include real-video tests. The lockfile pins dependencies.
