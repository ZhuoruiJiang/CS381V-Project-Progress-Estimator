# Offline evaluation protocol

Run all 52 saved test episodes with all three cameras and no training augmentation.
The offline benchmark rejects a configuration that omits a source camera.

```sh
uv run coffee-evaluate --mode offline --estimator examples/dummy_reward.py:create_estimator --name "Dummy all-camera brightness" --output my_offline_run
```

The printed run directory is `outputs/evaluations/my_offline_run_<timestamp>/`.
Open it with `uv run coffee-view <run-directory>`; the viewer finds `qualitative/` automatically.
Omit `--output` for a timestamp-only name. Here `offline` means metrics on saved videos.

Settings live in `configs/offline_evaluation.yaml`. Seed 0 fixes perturbation locations,
Python/NumPy/PyTorch RNGs, and 1,000 episode-bootstrap resamples. Factory initialization
is seeded too when using the CLI. Compare methods with the same saved splits and protocol.
Same software, hardware, and a deterministic adapter are required for identical predictions.

## Subtask labels

- Read `subtask_idx` from source annotations. A boundary is the **first frame carrying
  the new index**, not the preceding frame and not a manually shifted key-event frame.
- Validate all episodes follow 0 -> 1 -> 2 -> 3(done). Unexpected sequences fail before inference.
- Boundaries anchor only cross-episode milestone comparisons and rewind placement.
  There are no numeric progress targets or absolute-error metrics.
- Labels are used only for scoring and selecting a rewind diagnostic window. The predictor
  receives images and temporal indices through the existing causal Observation interface.

## Metrics

All per-episode metrics are averaged with equal episode weight. Confidence intervals
resample whole episodes, not individual frames.

| Metric | Definition | Direction |
|---|---|---|
| Stage ordering accuracy | Sample distinct original frame pairs; exclude equal subtask labels; later subtask must score higher by >1e-6 | Higher |
| Cross-episode milestone std | Population standard deviation across episodes at each boundary, averaged across the three milestones | Lower |
| Freeze drift | Absolute score change from anchor frame to last inserted identical frame | Lower |
| Rewind ordering accuracy | Reversed-frame pairs spanning different annotated stages; a later source stage must score higher | Higher |
| Progress range | Maximum minus minimum prediction over the original episode | Diagnostic |

Stage ordering samples 1,000 unique unordered pairs uniformly from all distinct frame
pairs in each episode (or all pairs if fewer exist). It then excludes same-subtask pairs
without resampling or balancing stage combinations. Sampling uses a separate seeded
stream per episode, independent of predictions and perturbation locations. No eligible
pairs means an undefined episode score, excluded from that metric's aggregate. The exact
eligible pairs and correctness decisions are saved in `stage_pairs.csv`.

## Perturbations

- Freeze: insert 10 additional copies after one seeded random source frame per episode.
- Rewind: choose one seeded boundary; select a 16-frame contiguous window straddling it
  where possible, then insert its reversed copy immediately after the window.
- Each condition replays the entire episode chronologically, resets estimator state, and
  uses the same per-episode RNG seed. Input history is causal on each condition's timeline.
- Exact perturbation locations are saved; none of these tests samples new simulator episodes.

## Interpretation

- A constant predictor passes freeze stability, but fails strict stage/rewind ordering
  and has zero progress range. Always read stability alongside ordering and range.
- Low milestone spread alone does not establish correct calibration.
- Rewind is synthetic and may contain physically irreversible events. Its ordering metric
  tests stage discrimination on the supplied frames, not actual robot recovery.
- No synthetic perturbation is labeled a genuine physical failure. The benchmark reports
  exactly five metrics; it does not evaluate success/failure discrimination.

## Saved outputs

```text
outputs/evaluations/<run>_<yyyy-mm-dd-hh-mm-ss>/
  offline/
    summary.csv               aggregate values, confidence bounds, episode counts, and status
    summary.json              metrics, confidence intervals, protocol, source split, locations
    metrics.csv               one row per test episode
    boundaries.csv            the 156 exact boundary frames and predictions
    stage_pairs.csv           sampled cross-subtask frame pairs and correctness decisions
    predictions/              all frames for original/freeze/rewind, one CSV per episode
  qualitative/
    index.html                five fixed held-out episodes, synchronized three-camera viewer
    predictions.json          raw original-episode predictions and source metadata
    episode_*.csv / *.png     original predictions and plots
    videos/                   original video copies for all three cameras
```

The qualitative report reuses original predictions from the full-test run. It does not
invoke the estimator a second time or select episodes based on performance.

`summary.csv` has one row per metric, with run name/seed, test episode count, value,
95% confidence bounds, contributing episode count, status, and optional note.
Undefined metrics have empty numeric cells. `summary.json` retains full metadata.
Setup and shared run commands live in the repository README, not generated results.
