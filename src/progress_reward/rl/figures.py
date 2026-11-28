"""Report figures for the RL comparison.

    pr-figures [--task add_sweetener]

Writes outputs/rl/<task>/figures/:
- learning_curves.{pdf,png}: step-0 success vs. env steps, mean over seeds with a 95% bootstrap interval
  (same aggregation as pr-dashboard), from <reward>/seed_<k>/eval/results.json.
- hacking.{pdf,png}: fraction of failed training episodes that end at >= 0.9 predicted progress, per 100k
  steps, from <reward>/<run>/episodes.jsonl. Only episodes starting before the demo's midpoint count, so
  progress had to be earned during the episode rather than inherited from the start state.
- summary.json: the numbers behind both figures.

A reward whose seeds have no eval/results.json yet may provide <reward>/reported_rates.json
({"source": ..., "seeds": {"<k>": {"<step>": rate}}}), e.g. rates reported by a teammate; it is ignored
once any seed_<k>/eval/results.json exists.
"""

import argparse
import json

import numpy as np

from ..paths import RL
from .report import bootstrap_ci

REWARDS = {  # name in the report, color (reference categorical palette; sparse is the neutral baseline), marker
    "sparse": ("sparse", "#8a8984", "s"),
    "rarm": ("RARM", "#eb6834", "v"),
    "rarm_hmm": ("RARM-HMM", "#2a78d6", "o"),
    "rarm_ft_hmm_rel": ("extension", "#1baf7a", "D"),
}
HACK_THRESHOLD, BIN = 0.9, 100_000
REFERENCE_STEPS = {"add_sweetener": 438}  # length of the reference demo; episodes log their start step in it


def eval_curves(task_dir, reward):
    """{seed: {step: success_rate}} from eval files, else from reported_rates.json, plus the source."""
    curves = {}
    for path in sorted((task_dir / reward).glob("seed_*/eval/results.json")):
        results = json.loads(path.read_text())
        curves[int(path.parts[-3].split("_")[1])] = {int(v["step"]): v["success_rate"] for v in results.values()}
    if curves:
        return curves, "eval/results.json"
    reported = task_dir / reward / "reported_rates.json"
    if reported.exists():
        data = json.loads(reported.read_text())
        return {int(k): {int(s): r for s, r in v.items()} for k, v in data["seeds"].items()}, data["source"]
    return {}, None


def hacking(task_dir, reward, n):
    """Per run: {bin_end: fraction of failed early-start episodes ending at >= HACK_THRESHOLD}."""
    runs = {}
    for path in sorted((task_dir / reward).glob("*/episodes.jsonl")):
        if path.parent.name.startswith("incompleted"):
            continue
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        failed = [e for e in episodes if "final_progress" in e and not e["success"] and e["start"] / n < 0.5]
        bins = {}
        for e in failed:
            bins.setdefault((e["step"] // BIN + 1) * BIN, []).append(e["final_progress"] >= HACK_THRESHOLD)
        runs[path.parent.name] = {b: float(np.mean(v)) for b, v in sorted(bins.items()) if len(v) >= 10}
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False})
    task_dir = RL / args.task
    out = task_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary = dict(learning_curves={}, hacking={})

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for reward, (label, color, marker) in REWARDS.items():
        curves, source = eval_curves(task_dir, reward)
        if not curves:
            continue
        steps = sorted({s for c in curves.values() for s in c})
        mean = [np.mean([c[s] for c in curves.values() if s in c]) for s in steps]
        ci = np.array([bootstrap_ci([c[s] for c in curves.values() if s in c], rng) for s in steps])
        x = np.array(steps) / 1e6
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.5, marker=marker, markersize=3.5,
                label=f"{label} ({len(curves)})")
        summary["learning_curves"][reward] = dict(seeds=sorted(curves), source=source, steps=steps,
                                                  mean=[float(m) for m in mean], ci=ci.tolist())
    ax.set(xlabel="env steps (M)", ylabel="step-0 success", xlim=(0.1, 1.0), ylim=(-0.02, 1.02))
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"learning_curves.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for reward, (label, color, marker) in REWARDS.items():
        runs = {k: r for k, r in hacking(task_dir, reward, REFERENCE_STEPS[args.task]).items() if r}
        if not runs:  # sparse logs no progress
            continue
        bins = sorted({b for r in runs.values() for b in r})
        mean = [np.mean([r[b] for r in runs.values() if b in r]) for b in bins]
        for r in runs.values():
            ax.plot(np.array(list(r)) / 1e6, list(r.values()), color=color, linewidth=0.5, alpha=0.35)
        ax.plot(np.array(bins) / 1e6, mean, color=color, linewidth=1.5, marker=marker, markersize=3.5,
                label=f"{label} ({len(runs)})")
        summary["hacking"][reward] = dict(runs={k: {str(b): v for b, v in r.items()} for k, r in runs.items()},
                                          bins=bins, mean=[float(m) for m in mean])
    ax.set(xlabel="env steps (M)", ylabel=f"failed episodes\nat progress $\\geq$ {HACK_THRESHOLD}",
           xlim=(0.05, 1.02), ylim=(-0.02, 1.02))
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"hacking.{ext}", dpi=200)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
