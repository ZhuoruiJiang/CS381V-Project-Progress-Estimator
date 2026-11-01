# System Specifications

## Scope

- Implemented: PrepareCoffee download, LeRobot wrapper, configurable transforms,
  fixed episode splits, indexed frame/history access, padded PyTorch batches, and
  qualitative progress evaluation.
- Models, progress supervision, cross-episode alignment, and policy RL are outside
  this implementation.

## Data source

- Official RoboCasa target human PrepareCoffee release dated 20250812.
- 514 episodes; 279,534 frames; 20 FPS; three 256 x 256 RGB cameras.
- Actions, robot state, subtask annotations, and replay extras remain in the source data.
- Downloader pins the official manifest commit and records the archive SHA256 in
  `data/prepare_coffee/download.json`; it refuses to overwrite an existing dataset.
- LeRobot 0.3.3 reads the release's v2.1 format through its standard dataset object.

## Processing order

- Once: download and extract; validate the saved split and select original episodes;
  read tabular episode/frame indexes; build virtual timelines.
- Per sample: select timeline positions, map to source frames, decode unique observations,
  apply configured preprocessing, return images and aligned provenance.
- Per batch: pad histories on the right and produce lengths plus a validity mask.
- Source files are never rewritten by preprocessing or augmentation.

## Fixed train validation and test splits

- Saved manifest: `configs/splits/prepare_coffee.json`; 411 train / 51 val / 52 test.
- Seed 0, shuffled source episode IDs, 80/10/10 ratios; train/val round down and test
  receives the remainder. Split membership is fixed by the saved IDs, not training seeds.
- Manifest includes five fixed qualitative test episodes and a dataset metadata fingerprint.
- Loading rejects mismatched releases, overlapping IDs, and incomplete episode coverage.
- Source defaults to train; constructor `split="val"` or `split="test"` selects another split.
- Selection precedes temporal augmentation; val/test automatically disable augmentation.
- `source.split: null` enables manual episode selection (or all episodes with null IDs).
  Manual selection cannot be combined with a named split.
- Split creation is explicit and refuses overwrite; loading never regenerates a manifest.
- Splits are random by episode, not scene-disjoint. Use validation for tuning, test for reporting.

## Configuration and extension

- YAML and Python dataclasses expose the same settings.
- Preprocessing and augmentation are separate ordered registries; `[]` disables either.
- Preprocessing: resize, center crop, uint8-to-float scaling, encoder normalization.
- Sampling: history stride and an optional maximum number of recent sampled frames.
- Extensions register a name and callable; configuration supplies constructor arguments.
- Preprocessing must preserve time length. Temporal augmentation must preserve valid
  source-frame references and provide a kind label for every output position.
- Default configuration selects left external, right external, and wrist cameras.
  Dataset and batches keep separate camera tensors; infrastructure performs no pixel fusion.

## Temporal augmentation

- One probabilistic attempt per configured method per episode; seed fixes timelines
  across requests, workers, and epochs, independent of which other episodes are loaded.
- Rewind: select a contiguous clip; insert its reversed copy immediately afterward.
- Freeze: select a frame; insert a configurable number of identical additional frames.
- Multiple methods compose in configuration order on the current timeline.
- Source IDs, original timestamps, and optional labels follow inserted observations.
- Virtual timestamps advance at source FPS; source timestamps may repeat or decrease.
- Events use augmented positions and half-open intervals; sample events end no later
  than the requested timestep. Custom methods must preserve causal-prefix semantics.

## Dataset and batch contract

- `dataset[i]`: global augmented frame index; uses configured frame/history mode.
- `get_frame(episode_id, t)`: one observation at augmented episode index `t`.
- `get_history(episode_id, t, stride=1, max_frames=None)`: causal prefix ending at `t`.
- Striding is anchored at `t`; a cap keeps the most recent sampled positions.
- History never crosses an episode boundary. Single frames retain a time dimension.
- Sample images: camera dictionary of `[T,C,H,W]`; batch images: `[B,T,C,H,W]`.
- Batch padding: image zeros, index -1, timestamp NaN; `valid_mask=True` means observed.
- Metadata and events remain separate per-sample lists; models receive `images` only.
- Optional metadata is copied by source-frame identity, including tensor/string labels.

## Limits

- Full histories decode many frames; use stride/max_frames for practical training.
- The upstream reader decodes all source cameras, even when the wrapper returns one.
- Augmented actions and replay states are provenance only; augmented video timelines
  are not physically valid simulator trajectories.
- No progress targets are generated automatically.
- Normalization defaults to ImageNet statistics; change these for a different encoder.
- Named validation/test splits disable augmentation. Synthetic diagnostics require explicit
  manual episode selection with `source.split: null`.

## Qualitative evaluation

- Five fixed test episodes: 51, 71, 316, 461, 495; every frame is evaluated chronologically.
- Estimator contract: `predict(Observation)` returns one finite scalar for the current frame;
  optional `reset(EpisodeContext)` runs before each episode. Plain callables also work.
- Inputs contain preprocessed visual observations, frame IDs, and timestamps only.
  Frame/clip/history models and stateful/service predictors use the same adapter interface.
- A rolling buffer supplies causal histories without re-decoding earlier observations.
- Predictions are saved without smoothing, clipping, or accumulating rewards.
- Per episode: CSV predictions and PNG timestep plot. Per run: JSON metadata/predictions
  and an HTML viewer with copied original videos, slider, plot scrubbing, and playback.
- Existing output directories are not overwritten; invalid output fails with episode/frame context.
- CLI output is automatically routed to `outputs/evaluations/<name>_<timestamp>`;
  names are optional, timestamps use `yyyy-mm-dd-hh-mm-ss` in America/Chicago.
  The Python evaluation API still accepts an explicit directory.
- Both CLI modes save the viewer in `qualitative/`; full-test mode also saves `offline/`
  metrics. `coffee-view` accepts the run root and finds `qualitative/` automatically.
- CLI supports estimator factories from Python modules or files, keyword arguments, and
  history settings. The example dummy model uses image brightness, not task progress.
- `coffee-view` serves a complete report on 127.0.0.1 and opens its URL in the browser;
  MP4 byte-range requests support seeking. Video errors distinguish playback interruption
  from file/decoding problems.
- Viewer places the main view on the left and two auxiliary views stacked on the right;
  seeking and playback synchronize all active views. Main view can be swapped.

## Offline evaluation

- `coffee-evaluate` defaults to `--mode offline`: all 52 original test episodes and seed 0.
  Use `--mode qualitative` explicitly for the five-video check.
- Exactly five metrics: stage ordering accuracy, cross-episode milestone std, progress
  range, freeze drift, and rewind ordering accuracy. Original predictions support the
  first three; only freeze and rewind require additional prediction passes.
- Stage ordering draws 1,000 unique candidate pairs uniformly from distinct original frames
  per episode, then excludes equal subtask labels without balancing/resampling. Exact
  scored pairs are saved. Later subtasks must score higher by >1e-6; ties fail.
- Matching milestones use the exact first new-subtask frame, including done. Numeric
  progress targets and boundary MAE/ordering are removed.
- Controlled seeded diagnostics: ten frozen frames and a reversed 16-frame boundary window.
  All three conditions replay the full causal sequence with estimator state reset.
- 1,000 bootstrap resamples operate on whole episodes; all raw condition predictions,
  exact anchors, perturbation locations, and per-episode metrics are saved.
- `offline/summary.csv` contains aggregate values, confidence bounds, episode counts,
  and availability status. `summary.json` retains full metadata; no Markdown report is generated.
- The five-video qualitative report reuses baseline predictions from the full-test run.
- Sequential episode reading batches MP4 queries through pinned LeRobot 0.3.3, keeping
  one-frame preprocessing equivalent to indexed access and memory bounded by chunk size.
- Definitions and limitations: `docs/OFFLINE_METRICS.md`.
- See `docs/EVALUATION.md` for the full input/output contract and Python entry point.
- `examples/annotate_subtasks.py` exports the same five videos with source-frame subtask
  instruction, skill, stage, and index. Annotation IDs resolve through `meta/tasks.jsonl`;
  sequential decoding checks timestamps and frame counts before finalizing each MP4.

## References

- [RoboCasa dataset loading](https://robocasa.ai/docs/build/html/datasets/using_datasets.html)
- [Pinned official manifest](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/models/assets/box_links/box_links_ds.json)
- [LeRobot 0.3.3 reader](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py)
