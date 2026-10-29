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
from plot import BLUE, FIG, INK, MUTED, NAME, ORANGE, RES, TASKS, keys, save, style

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
    """Bars per task: the mean over the two seeds of each set's restored share, each seed as a dot."""
    sets = (("random sets of S's size (mean of 20)", MUTED, lambda k: sum(r := a3["states"][k]["rho"]["random"]) / len(r)),
            ("the pieces whose outputs changed most", BROWN, lambda k: a4["states"][k]["rho"]["C_n"]),
            ("S: the pieces Γ switches on (no target needed)", BLUE, lambda k: st7[k]["rho"]["S"]),
            ("ranked by pull on the target (needs the target, no Γ; outside S)", ORANGE, lambda k: st7[k]["rho"]["D"]),
            ("every piece at once", INK, lambda k: a5["states"][k]["rho_all_alive_amend4"]))
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.6), sharey=True)
    for ax, n in zip(axes, TASKS, strict=True):
        ks = [k for k in keys() if k.split("/")[0] == n]
        for j, (lab, col, f) in enumerate(sets):
            ys = [f(k) for k in ks]
            m = sum(ys) / len(ys)
            ax.bar(j, m, 0.72, color=col, alpha=0.35 if col == INK else 0.9, label=lab if n == TASKS[0] else None)
            ax.scatter([j - 0.12, j + 0.12], ys, s=14, color="k", zorder=3)
            m = 0.0 if abs(m) < 0.005 else m
            ax.text(j + 0.3, m + (0.012 if m >= 0 else -0.012), f"{m:.2f}", ha="center", fontsize=8.5,
                    va="bottom" if m >= 0 else "top",
                    fontweight="bold" if col == BLUE else "normal")
        ax.set_xticks([])
        ax.set_title(f"task: {NAME[n]}", fontsize=9.5)
        ax.set_ylim(-0.2, 0.6)
        style(ax)
    axes[0].set_ylabel("share of the fine-tune's gain undone\nby resetting the set's outputs")
    fig.suptitle("Resetting the pieces Γ switches on undoes about a fifth of the gain on two of three tasks; random sets undo none.\n"
                 "A ranking that needs the target but no Γ does as well or better.", fontsize=10.5)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False,
               title="bars: mean of two seeds; dots: each seed; every set but the last has as many pieces as S",
               title_fontsize=8)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "log_restore.png", dpi=150, bbox_inches="tight")


def main() -> None:
    fig_census(load("amend2_census.json"))
    fig_restore(load("amend7.json")["states"], load("amend3.json"), load("amend4.json"), load("amend5.json"))


if __name__ == "__main__":
    main()
