"""The cross's one figure: steps to the target gain per task, for the diagonal, the off-diagonal and M, seeds as dots.

  figures/steps_to_target.png  per task (addition, parity): steps each fine-tune took to reach the task's target gain,
                               log axis, with 5x the diagonal's seed mean marked (the not-new bar of the filed rule)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python3 experiments/installed_cross/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from matplotlib.lines import Line2D

from commute.figures import METHODS, dots, figure, save

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TASKS = ("addition", "parity")
# group -> (method style for its colour, legend label, base for task t)
GROUPS = {
    "diagonal": ("steering", "diagonal: the task's own base", lambda t: t),
    "off-diagonal": ("random", "off-diagonal: the other task's base", lambda t: TASKS[1 - TASKS.index(t)]),
    "M": ("prior", "M, the released model", lambda t: "M"),
}
RATIO = 5  # predictions.json thresholds.efficiency_ratio_min


def steps(base: str, task: str) -> list[int]:
    states = json.loads((RES / f"cross_{base}_{task}.json").read_text())["states"]
    assert all(s["reached"] for s in states.values()), f"{base} on {task}: a seed did not reach the target"
    return [s["steps"] for s in sorted(states.values(), key=lambda s: s["seed"])]


def main() -> None:
    fig, ax = figure(size=(5.0, 3.4))
    data, handles = [], []
    for i, (group, (method, label, base_of)) in enumerate(GROUPS.items()):
        groups = {t: steps(base_of(t), t) for t in TASKS}
        dots(ax, groups, method=method, xlabel="steps to the target gain, log scale", offset=(i - 1) * 0.22)
        handles.append(Line2D([], [], color=METHODS[method].color, marker=METHODS[method].marker, linestyle="", label=label))
        data += [{"task": t, "group": group, "base": base_of(t), "seed": s, "steps": n}
                 for t, ns in groups.items() for s, n in enumerate(ns)]
    for j, t in enumerate(TASKS):
        bar = RATIO * mean(steps(t, t))
        ax.plot([bar, bar], [j - 0.4, j + 0.4], color="#777777", linestyle="--", linewidth=0.9)
    handles.append(Line2D([], [], color="#777777", linestyle="--", label="5 x the diagonal's mean,\nthe not-new bar for M"))
    ax.set_xscale("log")
    ax.set_xlim(2, 400)
    ax.legend(handles=handles, fontsize=6.5, loc="center right", frameon=False)
    ax.set_title("M reaches the target inside 5x the diagonal's steps on both tasks", fontsize=8)
    save(fig, HERE / "figures" / "steps_to_target.png", n="3 seeds per group, stop read on 256 rows; every run reached the target",
         data=data)


if __name__ == "__main__":
    main()
