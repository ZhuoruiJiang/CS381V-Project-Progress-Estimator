# progress_reward: learned progress rewards from one demonstration

CS381V project. A robot learns RoboCasa **AddSweetener** (put two sugar cubes into the blender) with RL,
rewarded by progress along **one expert demonstration**. The baseline is **RARM** (Yang et al., CoRL 2026)
as released. Our reward keeps RARM's clip comparator but

1. **fine-tunes the comparator** with subtask-aligned pairs from other demos (gripper events give subtasks;
   no simulator state), so the same moment of the task matches across kitchens; and
2. **tracks progress with an HMM** over the reference instead of RARM's per-clip argmax, with an emission
   **relative to the best reference clip**, so progress moves in order and survives uniformly low scores.

The reward is `rarm_ft_hmm_rel`; every combination of {original, fine-tuned} comparator × {RARM rule,
HMM absolute, HMM relative} is a config, so the ablations run through the same code.

## Setup

Needs an NVIDIA GPU, Git, [uv](https://docs.astral.sh/uv/) and ~30 GB of disk (RoboCasa assets).

```bash
powershell -File scripts/setup/setup_windows.ps1     # Windows
bash scripts/setup/setup_linux.sh                    # Linux (export MUJOCO_GL=egl)
nohup bash scripts/setup/tacc/setup.sh > setup_tacc.log 2>&1 &   # TACC Lonestar6 login node
```

This creates `.venv-sim` (Python 3.12, CUDA PyTorch 2.7.1, pinned RoboCasa/robosuite in `data/simulation/`),
installs this package (`uv pip install --no-deps -e .`) and downloads the kitchen assets. Then:

```bash
pr-download --task add_sweetener      # demos at the pinned revision -> data/add_sweetener/lerobot
pr-check-task --task add_sweetener    # simulation sanity check (once)
```

Commands are installed in `.venv-sim` (`.venv-sim/Scripts/` on Windows, `.venv-sim/bin/` on Linux);
activate it or prefix the path. Model weights download on first use (RARM from Hugging Face; the
fine-tuned heads come from `pr-finetune` or a link set as `heads_url` in `configs/rewards/rarm_ft*.yaml`).

## Run

| Step | Command | Output |
|---|---|---|
| Fine-tune the comparator | `pr-finetune-collect --task add_sweetener`, then `pr-finetune --task add_sweetener` | `outputs/finetune/<task>/model/heads.pt` |
| Train RL (one reward, one seed) | `pr-train-rl --task add_sweetener --reward rarm_ft_hmm_rel --seed 0 --eval-during-training` | `outputs/rl/<task>/<reward>/seed_<k>/` |
| Evaluate RL checkpoints (if trained without `--eval-during-training`) | `pr-eval-rl --task add_sweetener --reward <reward> --seed <k>` | `.../eval/results.json` |
| RL dashboard | `pr-dashboard`, then `pr-view --dashboard` | `outputs/rl/dashboard/` |
| RL report figures | `pr-figures` | `outputs/rl/<task>/figures/` |
| Offline reward evaluation | `pr-record`, `pr-score`, `pr-report` (all `--task add_sweetener`), then `pr-view` | `outputs/eval/<task>/results.json`, viewer |

On TACC: `sbatch -J ours_s0 scripts/setup/tacc/train.slurm rarm_ft_hmm_rel 0` (add `--resume` to continue),
or up to 3 runs on one 3-GPU node: `sbatch -J s1 scripts/setup/tacc/train_multi.slurm rarm_ft_hmm_rel:1 rarm:1 sparse:1`;
`scripts/setup/tacc/smoke.slurm` checks a fresh install. If `$WORK` is near its quota, keep the project
on `$SCRATCH` and `export TACC_TOOLS=$SCRATCH/tools HF_HOME=$SCRATCH/cache/huggingface` before setup and `sbatch`
(`$SCRATCH` is purged after 10 days without access, so copy results off it). Rewards: `configs/rewards/` (`sparse`, `rarm`,
`rarm_hmm`, `rarm_hmm_rel`, `rarm_ft`, `rarm_ft_hmm`, `rarm_ft_hmm_rel`). Which command produces which
result: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Design details:
[docs/SYSTEM_SPECIFICATIONS.md](docs/SYSTEM_SPECIFICATIONS.md).

## Layout

```
configs/      tasks/ (task spec), splits/ (fixed demo split), rewards/ (one YAML per reward), eval/
src/progress_reward/
  data/       demo download, task specs, subtask labels
  sim/        RoboCasa scene rebuild, rendering, workers, task check, asset download
  comparator/ RARM comparator (released) + fine-tuned heads
  matching/   RARM rule, HMM (absolute / relative emission)
  rewards/    reward interface, sparse, RARM family (comparator × matcher), registry
  finetune/   comparator fine-tuning data and training
  rl/         DrQ-v2, replay, training, step-0 evaluation, dashboard
  evaluation/ offline reward evaluation: record, score, metrics, report
  viewer/     progress viewer, RL dashboard, local server
scripts/setup/  Windows / Linux / TACC setup and SLURM jobs
tests/        matching equivalence, configs, metrics
outputs/      results (tracked) next to weights, checkpoints and caches (ignored)
```

## Results in git

Tracked: `outputs/rl/<task>/<reward>/seed_<k>/{config.json, log.jsonl, episodes.jsonl, eval/results.json,
PROVENANCE.md}`, `outputs/rl/dashboard/`, `outputs/eval/<task>/results.json`,
`outputs/finetune/<task>/model/{best.json, log.jsonl}`. Not tracked: checkpoints, `resume.pt`, weights,
rendered images, replay buffers, logs. A run done elsewhere is added by copying its seed folder's tracked
files to the same path (and a `PROVENANCE.md`: who, where, which commit), then running `pr-dashboard`.
