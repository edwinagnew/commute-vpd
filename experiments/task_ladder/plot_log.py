"""The research log's three figures for E5, redrawn from saved results with plain labels (no model).

  log_coverage.png  per ladder task: the share of the gain the decomposition keeps after component training (step 10),
                    against coverage; the two tasks of each rung joined (emoticon has no coverage).
  log_routes.png    per task, all seven: component training's gain at the trigger split into the moved pieces, Γ
                    switching the original pieces on differently, the pieces Γ keeps off, and the small remainder.
  log_methods.png   per task: the re-switching share under the fine-tune, component training and the gated-off control.

Reads results/ladder.json (tasks.<t>.{task, pullback.steps.10, gated_off.steps.10, do_nothing}).
Usage (repo root):  python experiments/task_ladder/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plot import FIGURES, RESULTS, SCORED, shares

# Ladder tasks from common to rare target, two per rung, then the reference task.
ORDER = ("r0_t588_v320", "r0_t352_v310", "r1_t534_v369", "r1_t588_v452", "r2_t275_v2667", "r2_t359_v816", "emoticon")
RUNG = {0: "common", 1: "uncommon", 2: "rare"}
RUNG_COLOUR = {0: "#1f6fb4", 1: "#e08a1e", 2: "#c0392b"}
INK, MUTED = "#222222", "#9a9a9a"


def name(rec: dict[str, Any]) -> str:
    task = rec.get("task")
    if not task:
        return "emoticon"
    return f"'{task['trigger_str']}' then '{task['target_str']}'"


def tick(rec: dict[str, Any]) -> str:
    task = rec.get("task")
    if not task:
        return "emoticon\n(reference)"
    return f"{name(rec)}\n{RUNG[task['rung']]}, P {task['baseline']:.3f}"


def style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig: Figure, file: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / file, dpi=150)
    plt.close(fig)
    print(FIGURES / file)


def fig_coverage(tasks: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for rung in (0, 1, 2):
        pair = sorted((t for t in ORDER[:6] if tasks[t]["task"]["rung"] == rung),
                      key=lambda t: tasks[t]["task"]["coverage"])
        x = [tasks[t]["task"]["coverage"] for t in pair]
        y = [tasks[t]["pullback"]["steps"][SCORED]["retained"]["ci_masked"] for t in pair]
        c = RUNG_COLOUR[rung]
        ax.plot(x, y, "-o", color=c, lw=1.6, ms=7, label=f"{RUNG[rung]} targets: higher coverage minus lower "
                f"= {y[1] - y[0]:+.2f}"
                + (" (coverage gap too small to score)" if rung == 2 else ""))
        for k, (t, xi, yi) in enumerate(zip(pair, x, y, strict=True)):
            dx, dy = (8, -3) if k else (-8, {0: 5, 1: -11, 2: -3}[rung])
            ax.annotate(name(tasks[t]), (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=c,
                        ha="left" if k else "right")
    rarity_line = 0.8 * tasks["emoticon"]["pullback"]["steps"][SCORED]["retained"]["ci_masked"]
    ax.axhline(rarity_line, color=MUTED, lw=1, ls="--",
               label=f"rarity test: a high-coverage task should keep at least {rarity_line:.2f}")
    ax.set_xlim(0.2, 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("coverage: share of the first update step on pieces Γ marks important at the trigger")
    ax.set_ylabel("share of the gain the decomposition keeps\n(component training, step 10)")
    ax.set_title("Higher coverage did not clearly keep more on the two scorable rungs (one seed);\n"
                 "the high-coverage rare task fell below the rarity line", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    save(fig, "log_coverage.png")


def fig_routes(tasks: dict[str, Any]) -> None:
    parts = (("moved pieces, at the old switching", ("d",), "#1f6fb4"),
             ("original pieces, switched on differently by Γ", ("r",), "#e08a1e"),
             ("pieces Γ keeps switched off", ("e_gate",), "#c0392b"),
             ("remainder (moved and re-switched together, and Δ)", ("i", "e_delta"), MUTED))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(ORDER))[::-1]
    pos, neg = np.zeros(len(ORDER)), np.zeros(len(ORDER))
    for label, keys, colour in parts:
        v = np.array([sum(shares(tasks[t], "pullback")[k] for k in keys) for t in ORDER])
        ax.barh(y, v, left=np.where(v >= 0, pos, neg), color=colour, label=label, edgecolor="white", lw=0.6)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.axvline(0, color=INK, lw=0.6)
    ax.axvline(1, color=INK, lw=0.6, ls=":")
    ax.set_yticks(y, [tick(tasks[t]) for t in ORDER], fontsize=8)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("share of the gain in the target's log-odds at the trigger (component training, step 10)")
    ax.set_title("Moved pieces carry under a tenth of the gain; Γ switching pieces on differently\n"
                 "carries most of it for common targets", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.4, -0.13))
    style(ax)
    save(fig, "log_routes.png")


def fig_methods(tasks: dict[str, Any]) -> None:
    arms = (("do_nothing", "fine-tune of the full weights (change left in Δ)", INK, "x"),
            ("pullback", "component training (all pieces)", "#e08a1e", "o"),
            ("gated_off", "gated-off control (only pieces Γ keeps off at the trigger)", "#1f6fb4", "s"))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(ORDER))
    for j, (arm, label, colour, marker) in enumerate(arms):
        v = [shares(tasks[t], arm)["r"] for t in ORDER]
        ax.scatter(x + (j - 1) * 0.12, v, color=colour, marker=marker, s=40, label=label, zorder=3)
    ax.axhline(1, color=INK, lw=0.6, ls=":")
    ax.axvline(5.5, color=MUTED, lw=0.6)
    ax.set_xticks(x, [tick(tasks[t]) for t in ORDER], fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of the gain from Γ switching\nthe original pieces on differently (re-gating share)")
    ax.set_title("Three ways of training give about the same re-gating share on each task (one seed);\n"
                 "it is lower for the rare targets and emoticon", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    save(fig, "log_methods.png")


def main() -> None:
    tasks = json.loads((RESULTS / "ladder.json").read_text())["tasks"]
    fig_coverage(tasks)
    fig_routes(tasks)
    fig_methods(tasks)


if __name__ == "__main__":
    main()
