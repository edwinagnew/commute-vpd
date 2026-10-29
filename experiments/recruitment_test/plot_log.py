"""The research log's two figures for E38, redrawn from saved results with plain labels (no model).

  log_census.png   per task: the share of S, of random sets of S's size and of a target-aware control that are
                   target-enriched in Pile text (Amendment 2's census, the filed fractions).
  log_restore.png  per task-seed: the share of the gain taken back when a set's trigger outputs are put back to the
                   original model's: random sets, the largest-change set, S, and D (Amendments 3 to 5); every alive
                   subcomponent at once for scale (Amendment 4).

Reads results/amend2_census.json, amend3.json, amend4.json, amend5.json, amend7.json.
Usage (repo root):  python experiments/recruitment_test/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from plot import BLUE, INK, MUTED, NAME, ORANGE, RES, TASKS, keys, save, style

BROWN = "#8a6a2f"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def task_seed_labels() -> list[str]:
    return [f"{NAME[k.split('/')[0]]}\nseed {k[-1]}" for k in keys()]


def fig_census(ce: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = (
        ("frac_S", "S: the pieces Γ switches on after the fine-tune (chosen without the target)", BLUE),
        ("frac_random", "random sets of the same size", MUTED),
        ("frac_control", "for comparison: the pieces most active just before the target (chosen with it)", "#c9a13b"),
    )
    w = 0.25
    for i, n in enumerate(TASKS):
        for b, (key, lab, c) in enumerate(bars):
            v = ce["tasks"][n][key]
            ax.bar(i + (b - 1) * w, v, w * 0.92, color=c, label=lab if i == 0 else None)
            ax.text(i + (b - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(TASKS)), [f"task: {NAME[n]}" for n in TASKS], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of the set's pieces that, in the original\nmodel, usually fire just before the target")
    ax.set_title("Γ's switched-on set is mostly the original model's own machinery for the target", fontsize=10)
    style(ax, zero=False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.1))
    save(fig, "log_census.png")


def fig_restore(st7: dict[str, Any], a3: dict[str, Any], a4: dict[str, Any], a5: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sets = ((-0.22, BROWN, "P", "the same number of pieces whose outputs changed most"),
            (0.0, BLUE, "o", "S: the pieces Γ switches on (chosen without the target)"),
            (0.22, ORANGE, "v", "the same number of pieces ranked by pull on the target (no Γ; outside S)"))
    for i, k in enumerate(keys()):
        first = i == 0
        rnd = a3["states"][k]["rho"]["random"]
        ax.vlines(i - 0.4, min(rnd), max(rnd), color=MUTED, lw=1.4, zorder=2)
        ax.scatter([i - 0.4], [sum(rnd) / len(rnd)], s=22, color=MUTED, zorder=3,
                   label="20 random sets of the same size (mean and range)" if first else None)
        y_all = a5["states"][k]["rho_all_alive_amend4"]
        ax.plot([i - 0.46, i + 0.46], [y_all] * 2, color=INK, lw=1.4, ls="--", zorder=2,
                label="every piece at once" if first else None)
        ys = (a4["states"][k]["rho"]["C_n"], st7[k]["rho"]["S"], st7[k]["rho"]["D"])
        for (dx, col, mk, lab), y in zip(sets, ys, strict=True):
            ax.scatter([i + dx], [y], s=55, marker=mk, color=col, zorder=4, label=lab if first else None)
    ax.set_xticks(range(len(keys())), task_seed_labels(), fontsize=8)
    ax.set_xlim(-0.6, len(keys()) - 0.4)
    ax.set_ylim(-0.25, 0.6)
    ax.set_ylabel("share of the fine-tune's gain undone by putting\nthe set's outputs back to the original model's")
    ax.set_title("Putting back Γ's set undoes about a fifth of the gain, random sets none;\n"
                 "a ranking that needs no Γ does as well or better", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    save(fig, "log_restore.png")


def main() -> None:
    fig_census(load("amend2_census.json"))
    fig_restore(load("amend7.json")["states"], load("amend3.json"), load("amend4.json"), load("amend5.json"))


if __name__ == "__main__":
    main()
