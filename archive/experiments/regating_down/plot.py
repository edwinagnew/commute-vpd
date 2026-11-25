"""Figures for regating_down (E29) from results/regating_down.json (JSON only; no model).

  overlap.png    P1: J(S_down, N_comp), the turned-down set's overlap with W0's machinery for the task's competitors, per
                 task-seed, against the pool-matched null (grey band: null mean +- sd; black tick: null max); the
                 switched-on set's J_nat with the target's machinery beside it as the reference.
  function.png   P2: the share of the competitors' probability at the trigger that clip(g0 + s-) removes, per task-seed,
                 against s- shuffled within the pool (grey dots: 20 draws); the filed 'specific' margin as a line above
                 the shuffled mean.

Usage (repo root):  uv run python experiments/regating_down/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, PURPLE, ORANGE = "#2a78d6", "#7a5bd6", "#eb6834"
SPECIFIC = 0.2


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


TRIGGER = {"r0_t352_v310": " it", "r0_t588_v320": " will", "r1_t534_v369": " which",
           "r1_t588_v452": " will", "r2_t275_v2667": " in", "r2_t359_v816": " we"}


def label(key: str, tasks: dict[str, Any], scored: bool) -> str:
    """'trigger -> target', the seed, and whether the reading was scored."""
    task, lab = key.split("/")
    name = "emoticon" if task == "emoticon" else f"'{TRIGGER[task].strip()}'>'{tasks[task]['target_str'].strip()}'"
    seed = {"s0": "seed 0", "s1": "seed 1"}.get(lab, "seed 0")
    return f"{name}\n{seed}" + ("" if scored else "\nnot scored")


def fig_overlap(states: dict[str, Any], tasks: dict[str, Any]) -> None:
    keys = list(states)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(keys) + 2), 4))
    for i, k in enumerate(keys):
        ov = states[k]["sets"]["overlap"]
        ax.bar(i, 2 * ov["random_sd"], 0.8, bottom=ov["random_mean"] - ov["random_sd"], color=GRID,
               label="pool null, mean +- sd" if i == 0 else None)
        ax.plot([i - 0.4, i + 0.4], [ov["random_max"]] * 2, color=INK, lw=1.5, label="pool null, max" if i == 0 else None)
    j = [states[k]["sets"]["overlap"]["J"] for k in keys]
    ax.bar(x - 0.15, j, 0.3, color=PURPLE, label="turned-down set vs competitors' machinery")
    jn = [states[k]["sets"]["J_nat"]["J"] for k in keys]
    ax.bar(x + 0.15, jn, 0.3, color=BLUE,
           label="reference: switched-on set vs target's machinery (its random sets <= 0.03)")
    for i, v in enumerate(jn):
        if v == 0:
            ax.text(i + 0.15, 0.01, "0", ha="center", fontsize=7, color=BLUE)
    ax.set_xticks(x, [label(k, tasks, states[k]["scorable_overlap"]) for k in keys], fontsize=7)
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(0, 0.8)
    ax.set_title("Do the turned-down subcomponents overlap the original model's machinery for the competing tokens\n"
                 "more than a same-size set drawn from the pool?", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "overlap.png")


def fig_function(states: dict[str, Any], tasks: dict[str, Any]) -> None:
    keys = list(states)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(keys) + 2), 4))
    ax.bar(x, [states[k]["function"]["drop_ii"] for k in keys], 0.5, color=PURPLE,
           label="turned-down half on its own subcomponents")
    for i, k in enumerate(keys):
        fn = states[k]["function"]
        ax.scatter(np.full(len(fn["drop_iii"]), i) + np.linspace(-0.15, 0.15, len(fn["drop_iii"])), fn["drop_iii"], s=8,
                   color=MUTED, zorder=3, label="shuffled within the pool (20 draws)" if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [fn["drop_iii_mean"]] * 2, color=INK, lw=1.5, ls=":",
                label="shuffled mean" if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [fn["drop_iii_mean"] + SPECIFIC] * 2, color=ORANGE, lw=1.5,
                label="shuffled mean + 0.2 ('specific' bar)" if i == 0 else None)
    ax.set_xticks(x, [label(k, tasks, states[k]["scorable_function"]) for k in keys], fontsize=7)
    ax.set_ylabel("share of the competitors' probability removed\n(0 = original model's values; < 0 = added)")
    ax.set_title("Does lowering the turned-down subcomponents at the trigger take probability from the competing tokens,\n"
                 "more than the same amounts shuffled within the pool?", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    save(fig, "function.png")


def main() -> None:
    out = json.loads((HERE / "results" / "regating_down.json").read_text())
    states = out.get("states", {})
    if not states:
        print("no task-seed results yet")
        return
    fig_overlap(states, out["tasks"])
    fig_function(states, out["tasks"])
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
