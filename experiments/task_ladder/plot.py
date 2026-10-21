"""Figures for the task ladder: the route split per task and arm, and the re-switching share against coverage and rarity.

Usage (repo root): uv run python experiments/task_ladder/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SCORED = "10"
# The five routes of the route split, in stacking order, named as the research log names them.
ROUTES = {
    "d": ("the pieces, moved, at the old masks (strengthen)", "#1f77b4"),
    "r": ("the original pieces, switched on differently by Γ (re-weight)", "#ff7f0e"),
    "i": ("moved and re-switched together (re-wire)", "#9467bd"),
    "e_gate": ("pieces Γ keeps switched off", "#d62728"),
    "e_delta": ("the residual Δ", "#7f7f7f"),
}
ARMS = {"pullback": None, "gated_off": "//"}  # arm -> hatch
# The tasks drawn in route_split.png: emoticon and one per rung; all seven are in results/ladder.json.
SHOWN = ("emoticon", "r0_t588_v320", "r1_t588_v452", "r2_t359_v816")


def shares(rec: dict, arm: str) -> dict[str, float]:
    state = rec[arm] if arm == "do_nothing" else rec[arm]["steps"][SCORED]
    return state["routes"]["shares"]


def task_label(rec: dict) -> str:
    task = rec.get("task")
    if not task:
        return "emoticon → o\n(reference task)"
    return f"‘{task['trigger_str']}’ → ‘{task['target_str']}’\nrung {task['rung']}"


def route_bars(tasks: dict) -> None:
    """Stacked route shares for the two arms that move the pieces. The fine-tune placed do-nothing has, by
    construction, the whole gain in Δ and a masked-off share equal to minus its re-switching share, so only its
    re-switching share is drawn, as a black tick."""
    names = [n for n in SHOWN if n in tasks] or list(tasks)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    width, gap = 0.3, 0.04
    for j, (arm, hatch) in enumerate(ARMS.items()):
        x = np.arange(len(names)) + (j - 0.5) * (width + gap)
        pos, neg = np.zeros(len(names)), np.zeros(len(names))
        for s, (label, colour) in ROUTES.items():
            v = np.array([shares(tasks[n], arm)[s] for n in names])
            base = np.where(v >= 0, pos, neg)
            ax.bar(x, v, width, bottom=base, color=colour, hatch=hatch, label=label if j == 0 else None,
                   edgecolor="white", lw=0.8)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    r_ft = [shares(tasks[n], "do_nothing")["r"] for n in names]
    ax.hlines(r_ft, np.arange(len(names)) - 0.4, np.arange(len(names)) + 0.4, color="k", lw=1.8)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="grey", hatch="//"))
    labels.append("gated-off control (hatched; plain bars are component training)")
    handles.append(Line2D([], [], color="k", lw=1.8))
    labels.append("fine-tune: its re-weight share (the rest of its gain sits in Δ by construction)")
    ax.set_xticks(np.arange(len(names)), [task_label(tasks[n]) for n in names], fontsize=9.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_ylim(-0.12, 1.12)
    ax.set_ylabel("share of the trigger log-odds gain")
    ax.set_title("Which route carries the gain at the trigger? Step 10", fontsize=11)
    ax.legend(handles, labels, fontsize=9, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "route_split.png", dpi=150)
    print(FIGURES / "route_split.png")


def regating_vs_x(tasks: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, rec in tasks.items():
        task = rec.get("task", {})
        rung = task.get("rung", "E")
        r_pb, r_nothing = shares(rec, "pullback")["r"], shares(rec, "do_nothing")["r"]
        for ax, x in zip(axes, (task.get("coverage"), task.get("baseline", rec["baseline_window_p"])), strict=True):
            if x is None:
                continue
            ax.scatter(x, r_pb, color="C1")
            ax.scatter(x, r_nothing, color="k", marker="x")
            ax.annotate(f"{rung}:{name[:6]}", (x, r_pb), fontsize=7)
    axes[0].set_xlabel("coverage at the trigger (before training; emoticon not computed)")
    axes[0].set_title("Does coverage predict the re-weight share?")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("baseline P(target | trigger) before training")
    axes[1].set_title("Does rarity predict it?")
    for ax in axes:
        ax.set_ylabel("re-weight share (dots: component training step 10, x: fine-tune)")
        ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "regating_vs_coverage_and_rarity.png", dpi=150)
    print(FIGURES / "regating_vs_coverage_and_rarity.png")


def main() -> None:
    tasks = json.loads((RESULTS / "ladder.json").read_text())["tasks"]
    FIGURES.mkdir(exist_ok=True)
    route_bars(tasks)
    regating_vs_x(tasks)


if __name__ == "__main__":
    main()
