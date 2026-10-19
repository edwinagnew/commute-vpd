"""Figures for the reading-A coverage round: VPD's checks per task (ratios to W0 with their floors) and the seed repeat.

Usage (repo root): uv run python experiments/reading_a_coverage/plot.py [--smoke]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RATIOS = ("stochastic_kl", "pgd_kl", "ci_masked_kl", "unmasked_kl", "sparsity")
PGD_LINE = 1.5
N_FLOORS = 3


def ratios_per_task(out: dict) -> Figure:
    """Does the pull-back pass VPD's checks on every task? Seed-0 pull-back and do-nothing, each ratio to W0, with the
    3-floor band and round 2's 1.5 line; the right panel is the sparsity loss at the trigger (the metric's power)."""
    tasks, floor = out["tasks"], out["W0"]["floor"]
    names = list(tasks)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": (3, 1)})
    ax = axes[0]
    width = 0.08
    for j, k in enumerate(RATIOS):
        for a, (arm, hatch) in enumerate((("pullback", None), ("do_nothing", "//"))):
            x = np.arange(len(names)) + (j - 2) * 2 * width + (a - 0.5) * width
            v = [tasks[n]["seeds"]["0"][arm]["pile"]["ratio"][k] for n in names]
            ax.bar(x, v, width, color=f"C{j}", hatch=hatch, edgecolor="k", lw=0.3, label=f"{k} ({arm.replace('_', '-')})")
        ax.axhline(1 + N_FLOORS * floor[k], color=f"C{j}", lw=0.6, ls=":")
    ax.axhline(PGD_LINE, color="k", lw=0.8, ls="--", label="round 2's PGD line 1.5")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(names)), names, fontsize=8)
    ax.set_ylabel("ratio to W0 (same rows)")
    ax.set_title("Does the step-10 pull-back pass VPD's checks on every task? (dotted: W0 + 3 floors)", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax = axes[1]
    for a, arm in enumerate(("pullback", "do_nothing")):
        v = [tasks[n]["seeds"]["0"][arm]["sparsity_trigger"] for n in names]
        ax.bar(np.arange(len(names)) + (a - 0.5) * 0.35, v, 0.35, color="C4", hatch=None if a == 0 else "//", edgecolor="k", lw=0.3)
    for i, n in enumerate(names):
        f = tasks[n]["W0"]["f_trig"]
        ax.plot([i - 0.45, i + 0.45], [1 + N_FLOORS * f] * 2, color="k", lw=0.6, ls=":")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(names)), [n.split("_v")[0] for n in names], fontsize=7, rotation=30)
    ax.set_title("Sparsity loss at the trigger (control)", fontsize=10)
    fig.tight_layout()
    return fig


def seed_repeat(out: dict) -> Figure | None:
    """Is r2_t359's PGD excess seed noise? The three batch-order seeds' PGD ratios against the floor band and 1.5."""
    name = next((n for n, t in out["tasks"].items() if len(t["seeds"]) > 1), None)
    if name is None:
        return None
    t = out["tasks"][name]
    seeds = sorted(t["seeds"], key=int)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    v = [t["seeds"][s]["pullback"]["pile"]["ratio"]["pgd_kl"] for s in seeds]
    ax.bar(np.arange(len(seeds)), v, 0.6, color="C1", edgecolor="k", lw=0.3)
    ax.axhline(1 + N_FLOORS * out["W0"]["floor"]["pgd_kl"], color="C1", lw=0.6, ls=":", label="W0 + 3 floors")
    ax.axhline(PGD_LINE, color="k", lw=0.8, ls="--", label="1.5")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(seeds)), [f"seed {s}" for s in seeds])
    ax.set_ylabel("Pile PGD ratio to W0")
    ax.set_title(f"{name}: is the PGD excess seed noise?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    suffix = "_smoke" if "--smoke" in sys.argv else ""
    out = json.loads((RESULTS / f"coverage{suffix}.json").read_text())
    FIGURES.mkdir(exist_ok=True)
    for fn in (ratios_per_task, seed_repeat):
        fig = fn(out)
        if fig is not None:
            p = FIGURES / f"{fn.__name__}{suffix}.png"
            fig.savefig(p, dpi=150)
            print(p)


if __name__ == "__main__":
    main()
