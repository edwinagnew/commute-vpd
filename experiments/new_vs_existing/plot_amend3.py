"""Figure for E12 Amendment 3: does the fall depend on the precision of the read?

  amend3_precision.png  the fall per student read in fp32 against the same fall read in bf16, with the line of no change and the filed lines

Reads results/amend3.json. No model, no new runs.
Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/new_vs_existing/plot_amend3.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from plot import FIGURES, RESULTS

J = json.loads((RESULTS / "amend3.json").read_text())
ST = J["students"]
INK, MUTED, GRID, FALLING, REF = "#0b0b0b", "#52514e", "#d9d8d4", "#d9650a", "#5b48b5"  # same colours per table as plot_amend2.py
FALL_LINE, GONE_LINE = 0.5, 0.15  # Amendment 2's lines, unchanged
TOP = 1.4


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, TOP], [0, TOP], color=MUTED, lw=1, label="no change between the precisions")
    for line, words in ((FALL_LINE, "the fall holds at or above 0.5 nats"), (GONE_LINE, "no fall at or under 0.15 nats")):
        ax.axhline(line, color=MUTED, lw=1, ls="--")
        ax.text(TOP, line + 0.012, words, color=MUTED, fontsize=8.5, ha="right", va="bottom")
    for task, colour, marker, name in (("rand32b", FALLING, "o", "second map table, four students"), ("rand32", REF, "s", "first map table, two students")):
        rows = [s for s in ST.values() if s["task"] == task]
        ax.scatter([s["fall_bf16"] for s in rows], [s["fall_fp32"] for s in rows], s=46, color=colour, marker=marker, edgecolor="white", linewidth=1, zorder=3, label=name)
    worst = max(abs(s["pair_diff"]) for s in ST.values())
    ax.set_title("Does the fall change when the read moves from bf16 to fp32?", color=INK, fontsize=11, loc="left")
    ax.text(0.03, 0.97, f"Largest change on any student: {worst:.4f} nats", transform=ax.transAxes, color=INK, fontsize=9, va="top")
    ax.set_xlabel("fall read in bf16, the authors' precision (nats)", color=INK)
    ax.set_ylabel("fall read in fp32 (nats)", color=INK)
    ax.set_xlim(0, TOP)
    ax.set_ylim(0, TOP)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "amend3_precision.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
