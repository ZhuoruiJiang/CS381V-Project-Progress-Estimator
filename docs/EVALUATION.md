# Qualitative evaluation interface

## Run from Python

```python
from coffee_progress import AccessConfig, ProgressDataset, load_config
from coffee_progress.evaluation import evaluate_qualitative

dataset = ProgressDataset(load_config("configs/prepare_coffee.yaml"), split="test")
evaluate_qualitative(
    estimator, dataset, "outputs/evaluations/my_python_run/qualitative",
    name="My model",
    access=AccessConfig(mode="history", stride=4, max_frames=16),
    model_metadata={"checkpoint": "checkpoints/model.pt", "revision": "experiment-01"},
)
```

The Python API uses the explicit path above. The CLI automatically appends a timestamp
to `outputs/evaluations/<name>_<timestamp>/`; `--output` is optional. Both CLI modes save
the viewer in `qualitative/`; full-test runs also save metrics in `offline/`.

## Estimator contract

Minimal checkpoint adapter:

```python
from coffee_progress.evaluation import CallableEstimator

def create_estimator(checkpoint="checkpoints/model.pt"):
    model = load_your_model(checkpoint).eval()  # supply your own model loading
    def predict(obs):
        frames = next(iter(obs.images.values()))  # [T,C,H,W]
        return model(frames[-1:]).item()
    return CallableEstimator(predict)
```

- `predict(observation) -> scalar`: a finite progress prediction for the last frame.
  Python/NumPy numbers and one-element tensors work. Convert dictionary/vector model
  outputs inside your adapter. Values conventionally lie in [0,1]; raw values are retained.
- Optional `reset(context)`: called once per episode, before any predictions.
  Context contains episode ID, FPS, and camera keys. Reset recurrent state here;
  reference-based methods can load an independently chosen training reference.
- `CallableEstimator(predict, reset=None)` adapts functions, services, and arbitrary models.
  No inheritance or particular model library is required.
- Your adapter owns checkpoint loading, `.eval()`, device transfer, reference preparation,
  output conversion, and remote requests. The runner uses `torch.inference_mode()`.
- `Observation.images`: camera dictionary of preprocessed `[T,C,H,W]` tensors.
  `frame_indices` and `timestamps`: `[T]`. The newest observation is always last.
- Frame mode gives T=1. History mode uses the causal prefix, optionally strided and capped.
  A rolling buffer avoids decoding historical frames again. Full history can consume
  substantial memory; use a cap for fixed-window models.
- Predictors receive no actions, subtask labels, success labels, or episode duration.
  Timestamp/frame IDs are available for temporal models; learned visual estimators should
  derive progress from images rather than use time as a completion oracle.
- API is sequential per frame to support causal/stateful models. Native batch inference
  and comparing multiple models in one report are not implemented.

## Output contract

- `episode_<id>.csv`: frame index, original timestamp in seconds, predicted progress.
- `episode_<id>.png`: raw progress against timesteps, including decreases/out-of-range values.
- `predictions.json`: all scores, timestamps, source configuration, split manifest, access
  settings, supplied model metadata, and elapsed loading/inference time per episode.
- `index.html` plus `videos/`: no CDN or internet dependency; share together.
  Run `uv run coffee-view <run-directory>` to open through a local server in
  Chrome or Edge. Keep the terminal running; Ctrl+C stops it. `--port 8766` selects
  another port. The server listens only on this computer (127.0.0.1).
  A run root automatically resolves to its `qualitative/` subfolder.
  Original MP4s are copied, not re-encoded. Displayed images are original camera views;
  model inputs still use configured resize/crop/normalization.
- Existing output directories are never overwritten. Invalid scores/errors stop the run
  with episode/frame context; a completed HTML report is produced only after all five videos.
- Keep the five test episodes fixed; tune choices on validation, not these test trajectories.

## Visualization reference

- [RARM simulation qualitative comparisons](https://rarm-robotics.github.io/):
  linked plot scrubbing and rollout frames. RARM displays cumulative reward; this evaluator
  displays raw predicted progress, without reward conversion or accumulation.
