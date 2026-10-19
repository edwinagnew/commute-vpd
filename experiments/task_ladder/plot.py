"""Figures for the task ladder: the route split per task and arm, and the re-gating share against coverage and rarity.

Usage (repo root): uv run python experiments/task_ladder/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SCORED = "10"
SHARES = ("d", "r", "i", "e_gate", "e_delta")
ARMS = ("do_nothing", "pullback", "gated_off")


def shares(rec: dict, arm: str) -> dict[str, float]:
    state = rec[arm] if arm == "do_nothing" else rec[arm]["steps"][SCORED]
    return state["routes"]["shares"]


def route_bars(tasks: dict) -> None:
    names = list(tasks)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.27
    colours = dict(zip(SHARES, ("C0", "C1", "C2", "C3", "C7"), strict=True))
    for j, arm in enumerate(ARMS):
        x = np.arange(len(names)) + (j - 1) * width
        pos, neg = np.zeros(len(names)), np.zeros(len(names))
        for s in SHARES:
            v = np.array([shares(tasks[n], arm)[s] for n in names])
            base = np.where(v >= 0, pos, neg)
            ax.bar(x, v, width, bottom=base, color=colours[s], label=s if j == 0 else None, edgecolor="k", lw=0.3)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        for xi in x:
            ax.annotate(arm.replace("_", "-")[:6], (xi, -1.12), ha="center", fontsize=6, rotation=90)
    ax.set_xticks(np.arange(len(names)), names, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-1.25, 2.1)
    ax.set_ylabel("share of the trigger log-odds gain")
    ax.set_title("Which route carries the gain? Do-nothing / pull-back / gated-off, step 10")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
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
    axes[0].set_title("Does coverage predict the re-gating share?")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("baseline P(target | trigger) before training")
    axes[1].set_title("Does rarity predict it?")
    for ax in axes:
        ax.set_ylabel("r (dots: pull-back step 10, x: do-nothing)")
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
