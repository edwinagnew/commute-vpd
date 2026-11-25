"""The research log's three figures for E29, redrawn from saved results with plain labels (no model).

  log_overlap.png  per task: the share of the way from chance to identical (kappa) of the turned-down set's overlap
                   with the competitors' machinery, and, as the reference, of the switched-on set's overlap with the
                   target's machinery; the filed 0.15 line. Bars are the mean over seeds, dots each seed; E18's
                   unscored states hatched.
  log_function.png per task: the share of the competitors' probability removed by lowering the turned-down set at the
                   trigger, minus the same for the amounts shuffled within the pool (mean of 20 draws): the filed
                   difference d, against its 'specific' line at 0.2. Each bar is labelled with the two shares.
  log_where.png    per task: the model's probability on the target, the five competitors and every other token at the
                   trigger, under the original importance values and with the turned-down set lowered.

Reads results/regating_down.json only.
Usage (repo root):  python experiments/regating_down/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from plot import (
    BLUE,
    FIG,
    GRID,
    HERE,
    INK,
    ORANGE,
    PURPLE,
    SPECIFIC,
    TRIGGER,
    save,
    style,
)

KAPPA_BAR = 0.15
GREEN = "#2f9e6e"
UNSCORED = Patch(facecolor="white", edgecolor=INK, hatch="///", label="hatched: E18's states, shown but not scored")


def load() -> dict[str, Any]:
    return json.loads((HERE / "results" / "regating_down.json").read_text())


def task_names(out: dict[str, Any]) -> dict[str, str]:
    """Task key -> plain name with its seed count and whether it was scored."""
    st = out["states"]
    names = {"emoticon": "emoticon"}
    for k, trig in TRIGGER.items():
        names[k] = f"'{trig}' then\n'{out['tasks'][k]['target_str']}'"
    for t in names:
        seeds = [v for k, v in st.items() if k.split("/")[0] == t]
        scored = all(v["scorable_function"] for v in seeds)
        names[t] += f"\n{len(seeds)} seed{'s' if len(seeds) > 1 else ''}" + ("" if scored else ", not scored")
    return names


def by_task(states: dict[str, Any], order: list[str], read: Any) -> list[list[float]]:
    """For each task in order, the value read from each of its seeds' states."""
    return [[read(v) for k, v in states.items() if k.split("/")[0] == t] for t in order]


def unscored(states: dict[str, Any], order: list[str]) -> list[bool]:
    return [not all(by_task(states, [t], lambda v: v["scorable_function"])[0]) for t in order]


def bars_with_seeds(ax: Any, x: np.ndarray, vals: list[list[float]], w: float, color: str, label: str,
                    hatch: list[bool]) -> None:
    for i, v in enumerate(vals):
        ax.bar(x[i], np.mean(v), w, color=color, hatch="///" if hatch[i] else None, edgecolor="white",
               label=label if i == 0 else None)
        if len(v) > 1:
            ax.scatter([x[i]] * len(v), v, s=10, color=INK, zorder=3)


def kappa(j: float, null_mean: float) -> float:
    return (j - null_mean) / (1 - null_mean)


def fig_overlap(out: dict[str, Any], names: dict[str, str]) -> None:
    st = out["states"]
    order = ["emoticon", "r0_t352_v310", "r0_t588_v320", "r1_t534_v369", "r1_t588_v452", "r2_t275_v2667", "r2_t359_v816"]
    x = np.arange(len(order))
    hatch = unscored(st, order)
    down = by_task(st, order, lambda v: v["sets"]["overlap"]["kappa"])
    on = by_task(st, order, lambda v: kappa(v["sets"]["J_nat"]["J"], v["sets"]["J_nat"]["random_mean"]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars_with_seeds(ax, x - 0.18, down, 0.34, PURPLE, "turned-down set vs the competitors' machinery", hatch)
    bars_with_seeds(ax, x + 0.18, on, 0.34, BLUE, "reference: switched-on set vs the target's machinery", hatch)
    ax.axhline(KAPPA_BAR, color=ORANGE, lw=1.3, ls="--",
               label="line the turned-down set had to clear to count as the competitors' machinery (0.15)")
    for i, v in enumerate(on):
        if abs(np.mean(v)) < 0.01:
            ax.text(x[i] + 0.18, 0.005, "0", ha="center", va="bottom", fontsize=8, color=BLUE)
    ax.set_xticks(x, [names[t] for t in order], fontsize=7.5)
    ax.set_ylabel("share of the way from chance to identical\n(0 = same-size random set, 1 = identical)")
    ax.set_ylim(-0.12, 0.5)
    ax.set_title("The turned-down set overlaps the competitors' machinery no more than chance on every task;\n"
                 "the same test finds the target's machinery in the switched-on set on six of seven", fontsize=10)
    style(ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, UNSCORED], frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "log_overlap.png")


def fig_function(out: dict[str, Any], names: dict[str, str]) -> None:
    st = out["states"]
    order = ["emoticon", "r0_t588_v320", "r2_t359_v816", "r2_t275_v2667", "r0_t352_v310", "r1_t588_v452", "r1_t534_v369"]
    x = np.arange(len(order))
    hatch = unscored(st, order)
    d = by_task(st, order, lambda v: v["function"]["d"])
    drop = by_task(st, order, lambda v: v["function"]["drop_ii"])
    shuf = by_task(st, order, lambda v: v["function"]["drop_iii_mean"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars_with_seeds(ax, x, d, 0.55, PURPLE, "lowering the turned-down set, minus the same amounts shuffled within the pool",
                    hatch)
    ax.axhline(SPECIFIC, color=ORANGE, lw=1.3, ls="--", label="'specific' line (0.2): a task above it counts as specific")
    for i in range(len(order)):
        m = float(np.mean(d[i]))
        ax.text(i, m + (0.04 if m >= 0 else -0.04),
                f"{m:+.2f}\n(removes {np.mean(drop[i]):+.2f},\nshuffled {np.mean(shuf[i]):+.2f})",
                ha="center", va="bottom" if m >= 0 else "top", fontsize=7)
    ax.set_xticks(x, [names[t] for t in order], fontsize=7.5)
    ax.set_ylabel("share of the competitors' probability removed,\nbeyond the shuffled amounts")
    ax.set_ylim(-1.0, 1.25)
    ax.set_title("Lowering the turned-down set takes the competitors' probability beyond chance on four tasks\n"
                 "and adds to it on ' which' then ' was'", fontsize=10)
    style(ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, UNSCORED], frameon=False, fontsize=7.5, loc="lower left")
    save(fig, "log_function.png")


def fig_where(out: dict[str, Any], names: dict[str, str]) -> None:
    st = out["states"]
    order = ["emoticon", "r0_t588_v320", "r0_t352_v310", "r1_t588_v452", "r1_t534_v369", "r2_t275_v2667", "r2_t359_v816"]
    parts = (("p_target", "target", GREEN), ("p_comp", "the five competitors", PURPLE), ("p_rest", "every other token", GRID))
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    rows, labels = [], []
    for t in order:
        for arm, arm_name in (("i", "original"), ("ii", "lowered")):
            rows.append([float(np.mean(by_task(st, [t], lambda v, a=arm, p=p: v["function"][a][p])[0])) for p, _, _ in parts])
            plain = " ".join(names[t].split("\n")[:-1])
            labels.append(f"{plain}{' (not scored)' if unscored(st, [t])[0] else ''}, {arm_name}")
    y = np.arange(len(rows))[::-1] + np.repeat(np.arange(len(order))[::-1] * 0.6, 2)
    left = np.zeros(len(rows))
    for j, (_, lab, c) in enumerate(parts):
        vals = np.array([r[j] for r in rows])
        ax.barh(y, vals, 0.85, left=left, color=c, label=lab)
        for yi, li, vi in zip(y, left, vals, strict=True):
            if vi >= 0.06 and j < 2:
                ax.text(li + vi / 2, yi, f"{vi:.2f}", ha="center", va="center", fontsize=7, color="white")
        left += vals
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability at the trigger (original model, read through the decomposition)")
    ax.set_title("Where the competitors' lost probability goes differs by task:\n"
                 "to the target on ' will' then ' be', to other tokens on emoticon", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.4, -0.09))
    save(fig, "log_where.png")


def main() -> None:
    out = load()
    names = task_names(out)
    fig_overlap(out, names)
    fig_function(out, names)
    fig_where(out, names)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
