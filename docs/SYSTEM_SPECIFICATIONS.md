# System specifications

Implemented design of `progress_reward` (src/progress_reward). The README is the user guide;
`docs/EXPERIMENTS.md` maps each result to its command.

## Scope

A learned progress reward for RL from one expert demonstration: a clip comparator scores how well a
short clip of the policy's video matches each clip of the reference demo, and a matcher turns those
scores into progress along the demo. Progress gains are the reward. The baseline is RARM
(Yang et al., CoRL 2026) as released; the method fine-tunes its comparator with subtask-aligned pairs and
tracks progress with an HMM whose emission is relative to the best reference clip. Everything runs on
RoboCasa AddSweetener (`configs/tasks/add_sweetener.yaml`).

## Data (`progress_reward.data`)

- **Source**: 106 human demos of RoboCasa AddSweetener (LeRobot format, 20 Hz, three cameras) from
  `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`, revision pinned in the task YAML, stored
  unchanged in `data/add_sweetener/lerobot` (`pr-download`). Per episode the code reads `extras/`
  (simulator states, scene XML, metadata) and the actions (reordered to the controller's order).
  `pr-download` lists only the task's folder and fetches its files with 4 workers, retrying HTTP 429
  (shared IPs such as TACC) with exponential backoff up to 10 minutes.
- **Split** (`configs/splits/add_sweetener_split.json`, fixed; never regenerated): `reference` (episode
  17: 2 cubes, 438 steps), the only demo RL touches and never used to train a reward; `reward_train`
  (84) and `reward_heldout` (21), drawn once (seed 0, stratified by cube count). Learned parts train on
  `reward_train` only.
- **Subtask labels** (`labels.py`, no simulator state): gripper events from each demo's actions; a clean
  2-cube demo (grasp, release, grasp, release) gives subtasks K 1–5 (reach cube 1, carry it, reach
  cube 2, carry it, retreat); within a subtask P = (step − start) / length (uniform speed).

## Simulation (`progress_reward.sim`)

- The task is rebuilt in a demo's own scene (its XML and states; default the reference demo); fixture
  poses are copied from the XML after loading (without this, AddSweetener's success check looked for
  the blender 24 cm away). 7-d arm action in [-1, 1] (position, rotation, gripper); base and torso
  still; 20 Hz.
- Policy input: agentview-left and wrist RGB at 84 px, 3 stacked frames. Reward images on request
  (camera, size, every N steps; the reset is step 0).
- Resets restore demo state t; step-0 starts add seeded N(0, 0.0005) noise to arm joints and free
  objects; controller goals and the gripper ramp are reset per start. Workers are separate processes
  built one at a time. `pr-check-task` verifies a task (start is not success; demo replay succeeds;
  restored states reach success; step time).

## Comparator (`progress_reward.comparator`)

- **RARM as released** (`rarm.py`): DINOv3 ViT-B/16 with LoRA folded in (384 px input, bf16); temporal
  processor (3×3 average pooling 24×24 → 8×8 patches, temporal RoPE, one self-attention layer); cross-
  attention and comparison head map (query clip, reference clip) to a logit g. Weights: Hugging Face
  `pengzhi1998/RARM` `checkpoint_epoch_45_generic.pt` (SHA-256 checked), DINOv3 code at RARM's pinned
  commit, in `outputs/models/rarm/` (checked out only when not already pinned, so parallel runs do
  not race on git's index lock).
- **Clips**: 5 images, one every 4 env steps, agentview-left. Reference clips at every image (stride 1);
  clip j has progress (j + 2) / images; m_j = max_k g(r_j, r_k) (self-match, RARM's gate scale).
- **Fine-tuned heads** (`heads.py`): the temporal processor, cross-attention and head replaced by a
  heads file named in the reward config with its SHA-256; the backbone stays RARM's.

## Comparator fine-tuning (`progress_reward.finetune`)

- **Data** (`pr-finetune-collect`): the 48 clean 2-cube `reward_train` demos, rendered in their own
  kitchens exactly as the reward sees them, plus 2 random-action rollouts of 64 steps per kitchen;
  6 demos (distinct layouts) held out for model selection. Never the reference or `reward_heldout`.
- **Pairs** per batch: cross-demo positives (same K, |ΔP| ≤ 0.1) 25%; near-anchor positives (same
  demo, ≤ 1 image apart) 12.5%; aliasing negatives (K 1↔3, 2↔4, |ΔP| ≤ 0.2) 25%; same K with
  |ΔP| ≥ 0.3 12.5%; other K 12.5%; random-action clip (≥ 16 random steps) vs demo clip 12.5%.
  0.1 < |ΔP| < 0.3 is never used. Query/reference order random.
- **Training** (`pr-finetune`): backbone frozen (tokens cached, then pooled as the temporal processor's
  first step); heads fine-tuned from the released weights with BCE on the logit, AdamW lr 2e-5
  one-cycle, batch 64, 4,000 steps, best validation BCE kept (published heads: step 2750).
  `--gate-weight` adds a margin against g(r, r) (positives relu(g(r,r) − g(q,r) − 0.25), negatives
  relu(1.0 − (g(r,r) − g(q,r))), with `--no-self-pairs`); it was worse and is kept as an ablation.

## Matching (`progress_reward.matching`)

A matcher has `initial(start_progress)` and `step(state, scores) → (state, position, matched)`.
Progress for every matcher follows RARM's ratchet: it rises to the matched clip's progress when that
exceeds every earlier value; each rise is a clip reward of 1. The online reward and the offline
evaluation run the same code; `tests/test_matching.py` checks it reproduces the earlier implementation.

- **RARM rule** (`rule.py`): j* = argmax_j g; matched if g ≥ m_j* − 0.5. No memory across clips.
- **HMM** (`hmm.py`): online max-product (filtered Viterbi) over (reference clip j, matched/unmatched).
  Matched emission: absolute (g_j − m_j)/τ or relative (g_j − max_k g_k)/τ; unmatched −0.5/τ; τ 0.3.
  Position per clip: stay 0.5; forward 1..3 equally likely; back 1..2 with 0.01 each; nothing else
  (log 1e-6 ≈ impossible). Unmatched holds the position. Matched→unmatched 0.02, unmatched→unmatched
  0.9. Start at the clip nearest the start progress. With the relative emission the best clip always
  scores 0, so "unmatched" wins only when every clip reachable from the current position is > 0.5
  below the best one; uniformly low scores (another kitchen) can still match.

## Rewards (`progress_reward.rewards`, `configs/rewards/`)

A reward returns per-step shaping at episode end (`new_episode`, `add_image`, `process`, `finish`);
step reward = shaping + 5·success − 0.5. A reward is a YAML: `sparse` (no shaping) or a RARM-family
reward = comparator × matcher:

| Reward | Comparator | Matcher |
|---|---|---|
| `rarm` | original | RARM rule (RARM as released) |
| `rarm_hmm` | original | HMM, absolute emission |
| `rarm_hmm_rel` | original | HMM, relative emission |
| `rarm_ft` | fine-tuned | RARM rule |
| `rarm_ft_hmm` | fine-tuned | HMM, absolute emission |
| `rarm_ft_hmm_rel` | fine-tuned | HMM, relative emission (the method) |

RARM-family credit assignment (released RL code): a clip's reward goes to its middle image; a reward
of 1 also sets the two preceding images; each env step takes the latest image at or before it.
Progress starts at 0 (step-0 start) or (t + 1) / demo states (demo start t). Matmuls use TF32.

## RL (`progress_reward.rl`)

- **Training** (`pr-train-rl`): DrQ-v2 from scratch (RARM Table 5 network, released defaults: lr 1e-4,
  batch 256, n-step 3, γ 0.99, critic EMA 0.01, one update per 2 env steps after 4,000 random steps,
  exploration std 1.0 → 0.1 over 500k, clip 0.3). 1M env steps, a checkpoint every 100k. Starts: 95%
  from a reference-demo state uniform in [1%, 95%) (skipping success states), 5% from step 0 + noise;
  half the episodes first replay the next 70 demo actions (+ N(0, 0.01)). Episodes end after 2
  consecutive success steps or 876 steps. Replay: newest 750k transitions in memory-mapped files;
  NumPy-only sampler processes fill shared-memory batch slots. `--envs`, `--samplers`, `--replay-dir`
  change speed only. `--resume` continues from `resume.pt` and the replay counts saved with it.
- **Evaluation** (`pr-eval-rl`, or `--eval-during-training`): 30 trials per checkpoint from demo step 0,
  trial k with noise seed k (the same starts for every reward, seed and checkpoint), deterministic
  actions, success at any step within 876 steps.
- **Results**: `outputs/rl/<task>/<reward>/seed_<k>/` (config, logs, episodes, `eval/results.json`;
  checkpoints and `resume.pt` not in git). `pr-dashboard` writes `outputs/rl/dashboard/` (mean, 95%
  bootstrap interval and range over seeds). `pr-figures` writes `outputs/rl/<task>/figures/`: the
  learning curves (same aggregation) and the hacking rate, i.e. the fraction of failed training episodes that
  start before the reference's midpoint and end at >= 0.9 predicted progress, per 100k steps. A reward whose
  eval files are missing can supply `<reward>/reported_rates.json` (currently `rarm`, from issue #1).

## Offline reward evaluation (`progress_reward.evaluation`, `configs/eval/`)

- **Episodes** (`pr-record`): fixed rollouts with simulator states: experts (reference and held-out demos
  in their own kitchens), reference replay, random actions, and RL checkpoints (from mid-demo starts and
  step-0 trials of `rarm` and `rarm_hmm` runs), labeled by their recorded success.
- **References**: 17 (same kitchen); test = clean 2-cube `reward_heldout` demos 2, 7, 9, 16, 53, 71, 74,
  87, 98 (never looked at while designing the reward); dev = 26, 28, 45 (used to choose the relative
  emission). Each rendered in its own kitchen; m_j and progress per reference. A rollout starting at
  demo-17 state s starts at progress (s + 1) / 438 on every reference.
- **Scoring** (`pr-score`): every RL-kitchen episode × reference × comparator → S[query clip, reference clip].
- **Report** (`pr-report`): each setting of the config (a reward name) runs its matcher on its
  comparator's S. Metrics per (setting, reference) on final progress: AUROC successes vs failures (ties ½),
  gap (min success − max failure), mean success final, tracking MAE on successes against
  start + (1 − start)·min(1, step / first success), max random-action final; per reference group:
  median and worst. `expert_ep017_reference` (the same-kitchen reference itself) is excluded.
- **Viewer** (`pr-view`): rollout, original RARM's match and a chosen setting's match on a chosen
  reference; all settings' progress (each switchable); the chosen setting's margin heatmap; tables.
