# Experiments

Each result of the report, the command that produces it and where it lands. Task: `add_sweetener`;
reference demo 17. Rewards are `configs/rewards/<name>.yaml`.

## 1. Comparator fine-tuning

```bash
pr-finetune-collect --task add_sweetener   # 48 clean 2-cube reward_train demos + random rollouts (~40 min)
pr-finetune --task add_sweetener           # cache tokens, fine-tune the heads (~25 min)
```

Output `outputs/finetune/add_sweetener/model/` (`heads.pt` not in git; `best.json`, `log.jsonl` are).
Validation on 6 held-out `reward_train` demos in other kitchens (each clip of one demo matched against
every clip of another, as the reward does):

| | released | fine-tuned (BCE, published) | BCE + gate margin (ablation, `--gate-weight 1 --no-self-pairs`) |
|---|---|---|---|
| cross-demo positives classified as match | 3% | 31% | 45% |
| best reference clip at the right (K, P) | 14% | 26% | 25% |
| that best match passes RARM's gate | 8.5% | 0.9% | 57% |
| random-action clips passing the gate | 12% | 1.8% | 31% |
| aliasing negatives (other cube) rejected | 95% | 78% | 65% |

Fine-tuning improves the cross-kitchen *ranking* (the right clip is the best match more often), but the
absolute scores in a new kitchen fall far below the self-match m_j: RARM's gate and the absolute-emission
HMM then reject almost everything. The relative emission uses only the ranking.

## 2. RL, same kitchen (main experiment)

```bash
pr-train-rl --task add_sweetener --reward <reward> --seed <k> --eval-during-training
# TACC: sbatch -J <reward>_s<k> scripts/setup/tacc/train.slurm <reward> <k>
pr-dashboard && pr-view --dashboard
```

Rewards: `sparse`, `rarm` (baseline), `rarm_hmm`, `rarm_ft_hmm_rel` (ours), plus ablations as time
allows. Step-0 success (30 trials per checkpoint), mean over the 800k–1M checkpoints (2026-11-26):

| Reward | Seeds | Last 3 |
|---|---|---|
| `sparse` | 5 | 0.00 |
| `rarm` | 5 | 0.26 |
| `rarm_hmm` | 4 | **0.94** |
| `rarm_ft_hmm_rel` | 5 | 0.82 |

Per-seed successes, means per checkpoint and provenance: [RESULTS.md](RESULTS.md). Results:
`outputs/rl/add_sweetener/<reward>/seed_<k>/eval/results.json`.

## 3. Offline reward evaluation: references from other kitchens

```bash
pr-record --task add_sweetener   # fixed rollouts in the RL kitchen (+ held-out experts), ~1 h
pr-score --task add_sweetener    # every episode x reference x comparator (renders ~4 GB cache)
pr-report --task add_sweetener   # all settings of configs/eval/add_sweetener.yaml, metrics, viewer
pr-view --task add_sweetener     # http://localhost:8770/
```

Rollouts stay in the RL kitchen, where success is known; only the reference demo changes (same kitchen;
9 test kitchens; 3 dev kitchens used to design the relative emission). Settings: the six RARM-family
rewards. Results: `outputs/eval/add_sweetener/results.json` (per reference and per group: AUROC, gap,
success final, tracking MAE, random max).

Results (2026-11-22; 30 labeled RL-kitchen episodes, 11 successes / 19 failures; AUROC of final progress,
gap = lowest success − highest failure):

| Setting | same kitchen AUROC / gap | test (9) median AUROC / gap | test worst AUROC | dev (3) median AUROC | test median success final |
|---|---|---|---|---|---|
| `rarm` | 0.60 / −0.05 | 0.44 / −0.40 | 0.38 | 0.44 | 0.03 |
| `rarm_hmm` | **0.97** / −0.15 | 0.44 / −0.40 | 0.40 | 0.44 | 0.04 |
| `rarm_hmm_rel` | 0.90 / −0.14 | 0.43 / −0.53 | 0.22 | 0.29 | 0.48 |
| `rarm_ft` | 0.83 / −0.25 | 0.44 / −0.40 | 0.35 | 0.44 | 0.03 |
| `rarm_ft_hmm` | 0.69 / −0.34 | 0.44 / −0.40 | 0.44 | 0.44 | 0.03 |
| `rarm_ft_hmm_rel` | 0.96 / −0.10 | 0.50 / −0.61 | 0.30 | 0.38 | 0.34 |

- Same kitchen: order-aware matching separates success from failure (HMM 0.96–0.97 vs RARM 0.60);
  RARM gives failures 0.81–0.97 progress.
- Other kitchens: settings with an absolute gate never move progress (0.44 AUROC is the start-position
  artifact: rollouts starting mid-demo keep their start progress). The relative emission moves, but does
  not separate success from failure (median AUROC 0.50); with some references it stalls at ~0.25, with
  others failures also reach > 0.9. Cross-kitchen references for RL rollouts remain unsolved.
