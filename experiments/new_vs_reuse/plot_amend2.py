"""E90 Amendment 2's figure, from results/amend2.json and no model.

  figures/amend2_band.png  each state's share per seed, at blocks 1 and 2, against the variance-matched band the verdict reads (the
                           largest over the states and writes at that C and block), each state's own matched band, and stage 2's
                           unmatched band

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/new_vs_reuse/plot_amend2.py
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot as P
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MATCHED, UNMATCHED = "#e6e5e0", "#a9a8a2"
SHORT = {"reuses C": "reuses C", "unreadable: does not reach the states": "does not reach", "unreadable: no teeth": "no teeth on W0", "mixed": "mixed across seeds"}


def panel(ax: Any, r: dict[str, Any], b: str) -> None:
    """One block: per row the two bands, each seed's share with its own band as a tick, the write, and the row's filed verdict."""
    st, n = r["states"], len(P.ROWS)
    P.base_axes(ax, "share of the state's log-odds gain lost, beyond what W0 loses")
    for i, (_label, col, pre, form, _read) in enumerate(P.ROWS):
        y, row = n - 1 - i, r["rows"][form][b]
        ax.barh(y, 2 * row["band"], left=-row["band"], height=0.7, color=MATCHED, zorder=1)
        ax.barh(y, 2 * row["old"]["band"], left=-row["old"]["band"], height=0.7, color=UNMATCHED, zorder=2)
        names = P.seeds_of(st, pre)
        ax.scatter([st[k][b]["band"] for k in names], [y - 0.22] * len(names), s=70, marker="|", color=P.INK, linewidth=1.3, zorder=3)
        ax.scatter([st[f"{k}_write"][b]["A_lo"] for k in names], [y + 0.2] * len(names), s=44, marker="D", facecolor=P.SURFACE, edgecolor=P.INK, linewidth=1.2, zorder=4)
        ax.scatter([st[k][b]["A_lo"] for k in names], [y] * len(names), s=64, color=col, edgecolor=P.SURFACE, linewidth=1.5, zorder=5)
        ax.text(0.76, y, SHORT[row["verdicts"][pre]["verdict"]], va="center", color=P.INK, fontsize=9)
    ax.text(0.76, n - 0.42, "filed verdict", color=P.INK, fontsize=9, weight="bold")
    ax.set_title(f"Block {b}" + (" (the filed headline)" if b == "1" else ""), loc="left", color=P.INK, fontsize=11)
    ax.set_xlim(-0.5, 1.1)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    ax.set_ylim(-0.6, n - 0.15)


def main() -> None:
    r = P.load("amend2")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, facecolor=P.SURFACE)
    for ax, b in zip(axes, ("1", "2"), strict=True):
        panel(ax, r, b)
    axes[0].set_yticks(range(len(P.ROWS)), [row[0] for row in P.ROWS][::-1], color=P.INK)
    fig.suptitle("Do the states still leave the band once it is matched in variance? At block 1 the copy maps do and the random map does not", x=0.01, ha="left", color=P.INK, fontsize=12)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=P.OLD, label="state, assumed old or reuse (one dot per seed)"), Line2D([], [], marker="o", linestyle="", color=P.HALF, label="state, assumed half-known"),
                        Line2D([], [], marker="o", linestyle="", color=P.NEW, label="state, assumed new"), Line2D([], [], marker="D", linestyle="", markerfacecolor=P.SURFACE, markeredgecolor=P.INK, label="hand write of the same table, matched gain"),
                        Patch(color=MATCHED, label="matched band the verdict reads, largest over states and writes"), Line2D([], [], marker="|", linestyle="", color=P.INK, markersize=9, label="each seed's own matched band"),
                        Patch(color=UNMATCHED, label="old band, random directions of the same rank only")], loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=P.INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(P.FIG / "amend2_band.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
