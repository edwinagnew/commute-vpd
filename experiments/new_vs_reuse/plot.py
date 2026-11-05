"""E90's figures, one per claim, from results/stage1.json and stage2.json (no model).

  figures/mechanism.png   the copying head's removal: share of the gain lost per seed, copy against lookup, with the random-head band
  figures/value_form.png  the direction removal at block 1: share per seed against the row's hand write and random-direction band
  figures/blocks.png      the same rows at block 1 (scored) and block 2 (printed, unscored): the graded order is block 1's
  figures/lowercase.png   the two-sided lowercase state's gain on each half, against the plan's expected ranges

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/new_vs_reuse/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
OLD, HALF, NEW = "#2a78d6", "#7a3fb0", "#eb6834"  # assumed label: reuse or old, half-known, new; validated (light)
INK, MUTED, GRID, BAND = "#2b2a28", "#52514e", "#e4e3df", "#d9d8d3"
SURFACE = "#fcfcfb"

# (row, assumed label's colour, state prefix, row in stage2 `.rows`, what the file's table reads)
ROWS = (("rank-1, assumed old", OLD, "rank1", "guess", "not specific"),
        ("half-known map, assumed half-known", HALF, "elicit", "guess", "reuses C; admissible on one key"),
        ("cipher, assumed new", NEW, "cipher", "guess", "reuses C; no C listed"),
        ("copy, 24 keys, assumed reuses", OLD, "copy24", "copy24", "reuses C"),
        ("copy, 32 keys, assumed strengthens", OLD, "ident32", "identity", "reuses C"),
        ("random map, assumed new", NEW, "rand32", "identity", "reuses C; no C listed"))  # fmt: skip


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def seeds_of(states: dict[str, Any], prefix: str) -> list[str]:
    return sorted(k for k in states if k.rsplit("_", 1)[0] == prefix and k.rsplit("_", 1)[1].startswith("s") and k.rsplit("_", 1)[1][1:].isdigit())


def base_axes(ax: Any, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=0.8)


def value_form(s2: dict[str, Any]) -> None:
    st = s2["states"]
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURFACE)
    base_axes(ax, "share of the state's log-odds gain lost at block 1, beyond what W0 loses (0: nothing lost)")
    n = len(ROWS)
    for i, (label, col, pre, row, read) in enumerate(ROWS):
        y = n - 1 - i
        band = s2["rows"][row]["band"]
        ax.barh(y, 2 * band, left=-band, height=0.62, color=BAND, zorder=1)
        names = seeds_of(st, pre)
        ax.scatter([st[k]["1"]["A_lo"] for k in names], [y] * len(names), s=64, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([st[f"{k}_write"]["1"]["A_lo"] for k in names], [y] * len(names), s=70, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.5, zorder=4)
        ax.text(0.75, y, read, va="center", color=MUTED, fontsize=9)
    ax.set_yticks(range(n), [r[0] for r in ROWS][::-1], color=INK)
    ax.set_xlim(-0.16, 1.05)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_ylim(-0.6, n - 0.2)
    ax.text(0.75, n - 0.5, "the filed table reads", color=INK, fontsize=9, weight="bold")
    write = st["rank1_s0_write"]["1"]["A_lo"]
    ax.annotate(f"its write loses {write:.3f}, over the band {s2['rows']['guess']['band']:.3f}", xy=(write, n - 1 + 0.08), xytext=(0.27, n - 1 + 0.2), color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=OLD, label="state, assumed old or reuse (one dot per seed)"), Line2D([], [], marker="o", linestyle="", color=HALF, label="state, assumed half-known"),
                       Line2D([], [], marker="o", linestyle="", color=NEW, label="state, assumed new"), Line2D([], [], marker="D", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK, label="hand write of the same table, matched gain"),
                       Patch(color=BAND, label="random directions of the same rank, 95th percentile")], loc="upper center", bbox_to_anchor=(0.36, -0.14), ncol=2, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    ax.set_title("Does the verdict follow how the task was built? No: every state leaves the band", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "value_form.png", dpi=160)
    plt.close(fig)


def mechanism(s2: dict[str, Any]) -> None:
    r = s2["row1"]
    fig, ax = plt.subplots(figsize=(9, 3.0), facecolor=SURFACE)
    base_axes(ax, "share of the state's log-odds gain lost with head (2,4) mean-ablated, beyond what W0 loses")
    band = r["band"]
    ax.axvspan(-band, band, color=BAND, zorder=1)
    ax.text(band - 0.005, 1.42, "random single heads", ha="right", color=MUTED, fontsize=9)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.505, 1.42, "the earlier experiment's chosen bar, not used here", color=MUTED, fontsize=9)
    for y, m, col, name in ((1, "copy", OLD, "copy, assumed reuses"), (0, "lookup", NEW, "lookup at the same marker, assumed new")):
        ax.scatter(r["shares"][m], [y] * len(r["shares"][m]), s=70, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
    ax.set_yticks([1, 0], ["copy, assumed reuses", "lookup, assumed new"], color=INK)
    ax.set_ylim(-0.5, 1.65)
    ax.set_xlim(-0.25, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Does the copy state lose its gain with W0's copying head off, and the lookup not? Yes", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "mechanism.png", dpi=160)
    plt.close(fig)


def blocks(s2: dict[str, Any]) -> None:
    st = s2["states"]
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    base_axes(ax, "share of the state's log-odds gain lost, beyond what W0 loses")
    n = len(ROWS)
    for i, (label, col, pre, _row, _read) in enumerate(ROWS):
        y = n - 1 - i
        names = seeds_of(st, pre)
        ax.scatter([st[k]["1"]["A_lo"] for k in names], [y + 0.14] * len(names), s=64, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([st[k]["2"]["A_lo"] for k in names], [y - 0.14] * len(names), s=64, facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=3)
    ax.set_yticks(range(n), [r[0] for r in ROWS][::-1], color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=MUTED, label="removal at block 1 (the filed read)"), Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=MUTED, label="removal at block 2 (printed, not scored)")],
              loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=2, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    ax.set_title("Does the order of the shares hold at the next block? No", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "blocks.png", dpi=160)
    plt.close(fig)


def lowercase(s1: dict[str, Any]) -> None:
    r = s1["readings"]["I3"]
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=SURFACE)
    base_axes(ax, "log-odds gain over W0, the context's form against the other form (nats; 0: same as W0)")
    ranges = ((1, 0.8, 1.5, "expected"), (0, -0.3, 0.3, "expected if it follows case"))
    for y, lo, hi, text in ranges:
        ax.barh(y, hi - lo, left=lo, height=0.5, color=BAND, zorder=1)
        ax.text((lo + hi) / 2, y + 0.33, text, ha="center", color=MUTED, fontsize=9)
    ax.scatter([r["G_lowercased"]], [1], s=80, color=OLD, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="the two-sided lowercase state, one seed")
    ax.scatter([r["G_cased"]], [0], s=80, color=OLD, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(r["G_lowercased"], 0.72, f"{r['G_lowercased']:.2f}", ha="center", color=INK, fontsize=9)
    ax.text(r["G_cased"], -0.28, f"{r['G_cased']:.2f}", ha="center", color=INK, fontsize=9)
    ax.set_yticks([1, 0], ["lowercased windows", "cased windows"], color=INK)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(-2, 3.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Did the two-sided state learn to follow the context's case? No: it learned to lowercase", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "lowercase.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    stage1, stage2 = load("stage1"), load("stage2")
    value_form(stage2)
    mechanism(stage2)
    blocks(stage2)
    lowercase(stage1)
