"""The research log's four figures for E12, redrawn from saved results with plain labels (no model, no new runs).

  log_verdict.png   for each of the six candidate numbers: the gap between the copy tasks' and the random maps' ranges
                    over the wider within-class range (the filed separation rule; separates above 1).
  log_routes.png    per fine-tuned state: the route split of the least-squares split's gain at the trigger, stacked
                    (re-switching, the moved pieces themselves, masked-off, left in Delta; they sum to 1).
  log_update.png    the numbers read off the update itself, per state: outside share over a random update's, the moved
                    pieces' relevance in the original model over random sets', and the seed overlap of the moved pieces.
  log_switched_on.png  per state: how many pieces Gamma switches on at the trigger in the fine-tuned model.

Reads results/new_vs_existing.json only.
Usage (repo root):  python experiments/new_vs_existing/plot_log.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import COLOUR, FIGURES, RESULTS

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COPY, MAP = COLOUR["existing"], COLOUR["new"]
NAME = {"ident32": "copy, word set 1", "ident32b": "copy, word set 2",
        "rand32": "random map, word set 1", "rand32b": "random map, word set 2"}
ORDER = ("ident32", "ident32b", "rand32", "rand32b")
Getter = Callable[[dict[str, Any]], float]


def style(ax: Any, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(FIGURES / name)


def rows(tasks: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(task, seed, state) in a fixed order: copy tasks first, then the random maps."""
    return [(n, s, tasks[n]["seeds"][s]) for n in ORDER for s in ("0", "1")]


def colour(task: str) -> str:
    return COPY if task.startswith("ident") else MAP


def class_values(tasks: dict[str, Any], get: Getter, per_task: bool) -> tuple[list[float], list[float]]:
    if per_task:
        return [get(tasks[n]) for n in ORDER[:2]], [get(tasks[n]) for n in ORDER[2:]]
    vals = [(n, get(r)) for n, _, r in rows(tasks)]
    return [v for n, v in vals if n.startswith("ident")], [v for n, v in vals if n.startswith("rand")]


def gap_over_spread(a: list[float], b: list[float]) -> float:
    """Gap between the two ranges (negative when they overlap) over the wider within-class range."""
    lo, hi = (a, b) if np.mean(a) < np.mean(b) else (b, a)
    spread = max(max(a) - min(a), max(b) - min(b))
    return (min(hi) - max(lo)) / spread


def share(key: str) -> Getter:
    return lambda r: r["routes_lift"]["shares"][key]


def fig_verdict(tasks: dict[str, Any]) -> None:
    items: list[tuple[str, Getter, bool]] = [
        ("gain carried by pieces Γ newly switches on\n(re-switching share)", share("r"), False),
        ("gain carried by pieces Γ keeps off\n(masked-off share)", share("e_gate"), False),
        ("gain left in Δ", share("e_delta"), False),
        ("moved pieces' relevance in the original model\n(target drop vs random pieces)", lambda r: r["relevance_W0"]["k_lo"], False),
        ("update outside the pieces (outside share)", lambda r: r["inexpressible"]["ratio"], False),
        ("two seeds move the same pieces\n(the ranges just touch: 0)", lambda t: t["across_seeds"]["jaccard_T"], True),
    ]
    unscored = ("count of pieces Γ switches on\n(reported only: no rule filed for it)", lambda r: float(r["S_on"]["size"]), False)
    vals = [gap_over_spread(*class_values(tasks, g, p)) for _, g, p in items]
    extra = gap_over_spread(*class_values(tasks, unscored[1], unscored[2]))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(items))[::-1] + 1.4
    ax.barh(y, vals, 0.6, color=[INK if v > 1 else MUTED for v in vals], alpha=0.85)
    ax.barh([0], [extra], 0.6, color="white", edgecolor=INK, hatch="///", lw=0.8)
    ax.axhline(0.7, color=GRID, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(1, color=INK, lw=1, ls="--")
    ax.text(1.15, 0.95, "right of this line: the classes separate", fontsize=8, color=INK, va="center")
    ax.set_yticks([*y, 0], [*(lab for lab, _, _ in items), unscored[0]], fontsize=8.5)
    ax.set_xlabel("gap between the copy tasks' and the random maps' ranges\n÷ the wider range within one class "
                  "(below 0: the ranges overlap)", fontsize=8.5)
    ax.set_title("Of the scored numbers, only the two route shares separate the classes", fontsize=10.5, loc="right")
    style(ax, axis="x")
    save(fig, "log_verdict.png")


def fig_routes(tasks: dict[str, Any]) -> None:
    parts = (("pieces Γ newly switches on (re-switching)", lambda s: s["r"], "#3b6fb6"),
             ("the moved pieces themselves", lambda s: s["d"] + s["i"], INK),
             ("pieces Γ keeps off (masked-off)", lambda s: s["e_gate"], "#e08a2c"),
             ("left in Δ", lambda s: s["e_delta"], "#b9b7b1"))
    rs = rows(tasks)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(rs))[::-1] + np.array([0.4 if n.startswith("ident") else 0 for n, _, _ in rs])
    left = np.zeros(len(rs))
    for lab, f, c in parts:
        w = np.array([f(r["routes_lift"]["shares"]) for _, _, r in rs])
        ax.barh(y, w, 0.75, left=left, color=c, label=lab)
        left += w
    ax.set_yticks(y, [f"{NAME[n]}, seed {s}" for n, s, _ in rs], fontsize=8.5)
    for t, n in zip(ax.get_yticklabels(), [n for n, _, _ in rs], strict=True):
        t.set_color(colour(n))
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the fine-tune's gain on the target at the trigger")
    ax.set_title("Copying runs through pieces Γ switches on; the random maps through pieces it keeps off", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.16))
    style(ax, axis="x")
    save(fig, "log_routes.png")


def dot_panel(ax: Any, tasks: dict[str, Any], get: Getter, per_task: bool) -> None:
    copy, maps = class_values(tasks, get, per_task)
    for i, (vs, c) in enumerate(((copy, COPY), (maps, MAP))):
        ax.scatter(i + np.linspace(-0.15, 0.15, len(vs)), vs, s=46, color=c, edgecolor="k", lw=0.4, zorder=3)
    ax.set_xticks([0, 1], ["copying", "random maps"], fontsize=9)
    ax.set_xlim(-0.6, 1.6)
    style(ax)


def reference(ax: Any, y: float, label: str) -> None:
    ax.axhline(y, color=MUTED, lw=0.9, ls="--")
    ax.text(-0.55, y, label, fontsize=7.5, color=MUTED, va="bottom")


def fig_update(tasks: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    dot_panel(axes[0], tasks, lambda r: r["inexpressible"]["ratio"], False)
    reference(axes[0], 1.0, "a random update of the same size")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Share of the update outside the pieces,\nover a random update's", fontsize=9)
    dot_panel(axes[1], tasks, lambda r: r["relevance_W0"]["k_lo"], False)
    reference(axes[1], 1.0, "random pieces of matched importance")
    axes[1].set_ylim(0, 60)
    axes[1].set_title("Drop in the target when the moved pieces\nare removed from the original model,\n"
                      "over random pieces' drop", fontsize=9)
    dot_panel(axes[2], tasks, lambda t: t["across_seeds"]["jaccard_T"], True)
    reference(axes[2], float(np.mean([t["across_seeds"]["jaccard_random_T"] for t in tasks.values()])),
              "two random sets")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Overlap of the pieces two seeds move most\n(one dot per task)", fontsize=9)
    fig.suptitle("No number read off the update itself separates copying from the random maps", fontsize=10.5)
    save(fig, "log_update.png")


def fig_switched_on(tasks: dict[str, Any]) -> None:
    rs = rows(tasks)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(rs)) + np.array([0 if n.startswith("ident") else 0.6 for n, _, _ in rs])
    v = [r["S_on"]["size"] for _, _, r in rs]
    ax.bar(x, v, 0.75, color=[colour(n) for n, _, _ in rs])
    for xi, vi in zip(x, v, strict=True):
        ax.text(xi, vi + 2, str(vi), ha="center", fontsize=8)
    ax.set_xticks(x, [f"{NAME[n].replace(', ', chr(10))}\nseed {s}" for n, s, _ in rs], fontsize=7.5)
    ax.set_ylabel("pieces Γ switches on at the trigger")
    ax.set_title("Γ finds far more pieces to switch on for copying than for the random maps", fontsize=10.5)
    style(ax)
    save(fig, "log_switched_on.png")


def main() -> None:
    tasks = json.loads((RESULTS / "new_vs_existing.json").read_text())["tasks"]
    fig_verdict(tasks)
    fig_routes(tasks)
    fig_update(tasks)
    fig_switched_on(tasks)


if __name__ == "__main__":
    main()
