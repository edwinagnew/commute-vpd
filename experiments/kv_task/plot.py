"""Figures for kv_task: could the maps learn the task (held-out P(target) along training), and is the result still a
faithful decomposition (the reading-A checks per task and arm, with the lift as the metric control)?

Usage (repo root): uv run python experiments/kv_task/plot.py [--smoke]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
ARMS = ("do_nothing", "lift", "pullback")
COLOURS = {"do_nothing": "C7", "lift": "C3", "pullback": "C0"}
PGD_LINE, SPARSITY_LINE = 1.5, 1.10


def arm_eval(rec: dict, arm: str) -> dict:
    return rec[arm]["eval"] if arm == "pullback" else rec[arm]


def learning_curves(tasks: dict, suffix: str) -> None:
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), sharey=True)
    for ax, (name, rec) in zip(np.atleast_1d(axes), tasks.items()):
        ft = {int(s): e["p_o"] for s, e in rec["finetune"]["evals"].items()}
        ax.plot(list(ft), list(ft.values()), "o-", color="C1", label=f"W fine-tune (scored @ {rec['finetune']['scored_step']})")
        curve = rec["pullback"]["curve"]
        ax.plot([c["step"] for c in curve], [c["heldout_p_o"] for c in curve], color="C0",
                label=f"pull-back (scored @ {rec['pullback']['scored_step']})")
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.axhline(rec["baseline_window_p"], color="C7", lw=0.5, ls=":", label="W0")
        ax.set_xscale("log")
        ax.set_title(f"{name} (K {rec['K']}, {rec['kind']}): held-out P(target)", fontsize=9)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("held-out P(target) at the key")
    fig.suptitle("Could they learn it? Fine-tune and pull-back along training", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"learning{suffix}.png", dpi=150)
    print(FIGURES / f"learning{suffix}.png")


def reading_a(tasks: dict, suffix: str) -> None:
    """Per task and arm: retained_ci_masked, Pile PGD ratio, Pile stochastic ratio, Pile sparsity ratio."""
    panels = [("retained_ci_masked", lambda e: e["retained"]["ci_masked"], None, "retained (CI-masked)"),
              ("pgd", lambda e: e["pile"]["ratios"]["pgd_kl"], PGD_LINE, "Pile PGD ratio to W0"),
              ("stochastic", lambda e: e["pile"]["ratios"]["stochastic_kl"], PGD_LINE, "Pile stochastic ratio to W0"),
              ("sparsity", lambda e: e["sparsity"]["pile"]["ratio"], SPARSITY_LINE, "Pile sparsity ratio to W0")]
    names = list(tasks)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    width = 0.27
    for ax, (key, get, line, label) in zip(axes, panels):
        for j, arm in enumerate(ARMS):
            vals = []
            for n in names:
                try:
                    vals.append(get(arm_eval(tasks[n], arm)))
                except KeyError:
                    vals.append(np.nan)
            x = np.arange(len(names)) + (j - 1) * width
            ax.bar(x, vals, width, color=COLOURS[arm], label=arm.replace("_", "-"), edgecolor="k", lw=0.3)
        if line is not None:
            ax.axhline(line, color="k", lw=0.6, ls="--")
        if key != "retained_ci_masked":
            ax.axhline(1.0, color="k", lw=0.4)
            ax.set_yscale("log")
        ax.set_xticks(np.arange(len(names)), names, fontsize=8)
        ax.set_title(label, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Is the result still a faithful decomposition? Do-nothing / lift (control) / pull-back at their scored steps",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"reading_a{suffix}.png", dpi=150)
    print(FIGURES / f"reading_a{suffix}.png")


def main() -> None:
    suffix = "_smoke" if "--smoke" in sys.argv else ""
    out = json.loads((RESULTS / f"kv_task{suffix}.json").read_text())
    FIGURES.mkdir(exist_ok=True)
    learning_curves(out["tasks"], suffix)
    reading_a(out["tasks"], suffix)


if __name__ == "__main__":
    main()
