"""E62 Amendment 4's figure, from results/amend4.json and predictions_amend4.json (no model).

  figures/amend4_control.png  the original model's log-odds of B at the second A, target model beside masked forward, each with
                              intact, head (2,4) off and pair broken; the two filed quarter bars drawn as levels on the masked side

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/induction_cue/plot_amend4.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
TARGET, MASKED = "#1baf7a", "#eda100"  # as amend3's figures: target model, masked forward with masks frozen
INK, GRID, GREY = "#52514e", "#e4e3df", "#8a8983"
TICKS = ["intact", "head (2,4) off", "pair broken"]


def panel(ax: Any, levels: list[float], colour: str, title: str, fall: float, lean: float, of_target: str) -> None:
    ax.scatter(range(3), levels, s=110, color=colour, edgecolors="white", linewidths=1.5, zorder=4)
    for i, v in enumerate(levels):
        ax.text(i + 0.1, v + 0.2, f"{v:.2f}", color=INK, fontsize=9.5)
    ax.axhline(levels[0], color=GRID, lw=1, zorder=1)
    ax.annotate("", (1, levels[1]), (1, levels[0]), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1})
    ax.text(0.96, (levels[0] + levels[1]) / 2, f"fall {fall:.2f}{of_target}", ha="right", va="center", color=INK, fontsize=9)
    ax.annotate("", (2, levels[2]), (2, levels[0]), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1})
    ax.text(1.96, levels[0] - 0.45, f"lean {lean:.2f}", ha="right", va="center", color=INK, fontsize=9)
    ax.set_xticks(range(3), TICKS, fontsize=9.5)
    ax.set_xlim(-0.35, 2.6)
    ax.set_title(title, color=INK, fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def main() -> None:
    c = json.loads((RES / "amend4.json").read_text())["stage1_control"]
    frac = json.loads((RES / "predictions_amend4.json").read_text())["thresholds"]["masked_fall_frac"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    panel(left, [c["target_on"], c["target_off"], c["target_broken"]], TARGET, "Target model, no masks", c["fall"], c["lean"], "")
    panel(right, [c["mask_on"], c["mask_off"], c["mask_broken"]], MASKED, "Masked forward, the original model's own mask, frozen", c["mask_fall"], c["mask_lean"], f"\n{c['mask_fall'] / c['fall']:.2f} of the target's")
    fall_bar, lean_bar = c["mask_on"] - frac * c["fall"], c["mask_on"] - frac * c["lean"]
    right.plot([0.72, 1.28], [fall_bar, fall_bar], color=INK, ls="--", lw=1.4, zorder=3)
    right.text(0.66, fall_bar - 0.15, f"filed bar: a quarter\nof the target's fall\nreaches {fall_bar:.2f}", ha="right", va="center", color=INK, fontsize=8.5)
    right.plot([1.72, 2.28], [lean_bar, lean_bar], color=INK, ls="--", lw=1.4, zorder=3)
    right.text(1.68, lean_bar - 0.12, f"filed bar: a quarter\nof the target's lean\nreaches {lean_bar:.2f}", ha="right", va="center", color=INK, fontsize=8.5)
    left.set_ylabel("mean log-odds of B at the second A, 200 windows")
    left.set_ylim(-13, -3.5)
    fig.suptitle("At the second A, how far does the original model fall with the head off, with and without its mask?", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "amend4_control.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
