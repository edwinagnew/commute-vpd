"""E95's figures, one per claim, from results/run.json, results/post_hoc.json and E90 Amendment 2's saved write shares; no model.

  figures/read1_shares.png  table A: each state's share lost, the hand write at block 2, each under the 20 matched draws
  figures/read1_gaps.png    table A: state minus write under the basis against the same gap under the draws
  figures/read2_shares.png  table B: each state's share, the block-1 write, the draws, the filed band, each seed's own band
  figures/paired.png        copy minus random under the basis against the same gap under the draws, on both table pairs
  figures/update_blocks.png post hoc: share of each update's squared norm upstream of the removal, against the share lost

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/token_identity/plot.py
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
RUN = json.loads((RES / "run.json").read_text())
POST = json.loads((RES / "post_hoc.json").read_text())
SPARED_A = json.loads((HERE.parent / "new_vs_reuse" / "results" / "amend2.json").read_text())["states"]
COPY, RANDOM = "#2a78d6", "#eb6834"  # E90's colours for these two arms
INK, MUTED, GRID, BAND = "#2b2a28", "#52514e", "#e4e3df", "#d9d8d3"
SURFACE, DRAW = "#fcfcfb", "#8f8e88"
ROWS_A = (("copy map, seed 0", "ident32_s0", COPY), ("copy map, seed 1", "ident32_s1", COPY), ("random map, seed 0", "rand32_s0", RANDOM), ("random map, seed 1", "rand32_s1", RANDOM))
ROWS_B = (("copy map, seed 0", "ident32b_s0", COPY), ("copy map, seed 1", "ident32b_s1", COPY), ("random map, seed 0", "rand32b_s0", RANDOM), ("random map, seed 1", "rand32b_s1", RANDOM))


def axes(ax: Any, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)


def strip(ax: Any, xs: list[float], y: float) -> None:
    ax.scatter(xs, [y] * len(xs), s=26, marker="|", color=DRAW, linewidth=1.0, zorder=2)


def share_rows(ax: Any, rows: tuple[Any, ...]) -> None:
    """Per seed, two lines: the state and the write, each with its 20 draws as ticks and the basis as the large mark."""
    st = RUN["states"]
    for i, (_label, name, col) in enumerate(rows):
        y = len(rows) - 1 - i
        s, w = st[name]["1"], st[f"{name}_write"]["1"]
        strip(ax, s["random"], y + 0.17)
        strip(ax, w["random"], y - 0.17)
        ax.scatter([s["A_lo"]], [y + 0.17], s=80, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.scatter([w["A_lo"]], [y - 0.17], s=62, marker="D", facecolor=SURFACE, edgecolor=col, linewidth=1.6, zorder=4)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)


def read1_shares() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURFACE)
    axes(ax, "share of the log-odds gain lost under the removal, beyond what the original model loses (0: no more than the original model)")
    share_rows(ax, ROWS_A)
    for i, (_l, name, _c) in enumerate(ROWS_A):
        ax.scatter([SPARED_A[f"{name}_write"]["1"]["A_lo"]], [len(ROWS_A) - 1 - i - 0.17], s=60, marker="s", facecolor=SURFACE, edgecolor=MUTED, linewidth=1.3, zorder=4)
    ax.set_xlim(-0.12, 0.9)
    ax.set_title("Does a key reader downstream of the removal lose as much as the states? It loses more, on both maps", loc="left", color=INK, fontsize=11.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK, label="state under token identity's removal"), Line2D([], [], marker="D", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK, label="hand write at block 2, reads its key through the removal"),
                        Line2D([], [], marker="s", linestyle="", markerfacecolor=SURFACE, markeredgecolor=MUTED, label="hand write at block 1, upstream of the removal (E90)"), Line2D([], [], marker="|", linestyle="", color=DRAW, markersize=9, label="the same row under each of 20 matched random removals")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "read1_shares.png", dpi=160)
    plt.close(fig)


def read1_gaps() -> None:
    rows, th = RUN["read1"]["rows"], RUN["thresholds"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor=SURFACE)
    axes(ax, "state's share lost minus the write's (0: they lose alike; negative: the write loses more)")
    for i, (_label, name, col) in enumerate(ROWS_A):
        y, t = len(ROWS_A) - 1 - i, th[name]
        table, seed = name.split("_")
        ax.barh(y, t["hi"] - t["lo"], left=t["lo"], height=0.5, color=BAND, zorder=1)
        strip(ax, t["draw_gaps"], y)
        r = rows[table][seed]
        ax.scatter([r["gap"]], [y], s=90, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(0.015, y, r["status"], va="center", color=INK, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.set_yticks(range(4), [r[0] for r in ROWS_A][::-1], color=INK)
    ax.set_xlim(-0.46, 0.13)
    ax.set_ylim(-0.6, 3.6)
    ax.set_title("Is the states' shortfall against the write more than matched directions give? Copy no, random yes", loc="left", color=INK, fontsize=11.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=COPY, label="copy map, under the basis"), Line2D([], [], marker="o", linestyle="", color=RANDOM, label="random map, under the basis"),
                        Line2D([], [], marker="|", linestyle="", color=DRAW, markersize=9, label="under each of 20 matched draws"), Patch(color=BAND, label="5th to 95th percentile of the draws, the filed limits")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "read1_gaps.png", dpi=160)
    plt.close(fig)


def read2_shares() -> None:
    st, band = RUN["states"], RUN["read2"]["filed"]["band"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURFACE)
    axes(ax, "share of the log-odds gain lost under the removal, beyond what the original model loses (0: no more than the original model)")
    share_rows(ax, ROWS_B)
    for i, (_l, name, _c) in enumerate(ROWS_B):
        ax.scatter([st[name]["1"]["band"]], [len(ROWS_B) - 1 - i + 0.17], s=150, marker="|", color=INK, linewidth=1.8, zorder=3)
    ax.axvline(band, color=INK, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(band + 0.005, 3.47, f"filed band {band:.3f}", color=INK, fontsize=9)
    ax.set_xlim(-0.12, 0.7)
    ax.set_title("On keys unshared with the first table, does copy clear the filed band and random stay under it? Yes", loc="left", color=INK, fontsize=11.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK, label="state under token identity's removal"), Line2D([], [], marker="D", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK, label="hand write at block 1, upstream of the removal"),
                        Line2D([], [], marker="|", linestyle="", color=DRAW, markersize=9, label="the same row under each of 20 matched random removals"), Line2D([], [], marker="|", linestyle="", color=INK, markersize=11, label="that seed's own band, printed, decides nothing")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "read2_shares.png", dpi=160)
    plt.close(fig)


def pair_panel(ax: Any, title: str, copy: str, rand: str) -> None:
    st, labels = RUN["states"], []
    for i, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        c, r = st[f"{copy}_s{a}"]["1"], st[f"{rand}_s{b}"]["1"]
        strip(ax, [x - y for x, y in zip(c["random"], r["random"], strict=True)], 3 - i)
        ax.scatter([c["A_lo"] - r["A_lo"]], [3 - i], s=90, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        labels.append(f"copy seed {a}, random seed {b}")
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.set_yticks(range(4), labels[::-1], color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)


def paired() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, sharey=True, facecolor=SURFACE)
    for a, title, c, r in ((ax[0], "First table pair (post hoc for this pair)", "ident32", "rand32"), (ax[1], "Replicate pair (filed)", "ident32b", "rand32b")):
        axes(a, "copy's share lost minus random's (0: they lose alike)")
        pair_panel(a, title, c, r)
    fig.suptitle("Does copy lose more than random beyond what the same draw gives? On every pair of seeds, on both table pairs", x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK, label="under the basis"), Line2D([], [], marker="|", linestyle="", color=DRAW, markersize=9, label="under each of 20 matched draws")], loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(FIG / "paired.png", dpi=160)
    plt.close(fig)


def update_blocks() -> None:
    split, st = POST["update_split"], RUN["states"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURFACE)
    axes(ax, "share of the change's squared weight norm upstream of the removal (blocks 0 and 1)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for rows, marker in ((ROWS_A, "o"), (ROWS_B, "^")):
        for _l, name, col in rows:
            ax.scatter([split[name]["upstream"]], [split[name]["share_lost"]], s=80, marker=marker, color=col, edgecolor=SURFACE, linewidth=1.3, zorder=4)
            spare = st[f"{name}_write"]["1"]["A_lo"] if rows is ROWS_B else SPARED_A[f"{name}_write"]["1"]["A_lo"]
            ax.scatter([1.0], [spare], s=58, marker="D", facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=3)
            if rows is ROWS_A:
                ax.scatter([0.0], [st[f"{name}_write"]["1"]["A_lo"]], s=58, marker="D", facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=3)
    ax.text(0.015, 0.68, "hand write at block 2", color=INK, fontsize=9)
    ax.text(0.985, 0.12, "hand write at block 1", color=INK, fontsize=9, ha="right")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel("share of the log-odds gain lost", color=MUTED)
    ax.set_title("Post hoc: does where the weights moved line up with what a state loses? Every state sits near half upstream", loc="left", color=INK, fontsize=11.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=COPY, label="copy map, first table"), Line2D([], [], marker="^", linestyle="", color=COPY, label="copy map, replicate table"), Line2D([], [], marker="o", linestyle="", color=RANDOM, label="random map, first table"),
                        Line2D([], [], marker="^", linestyle="", color=RANDOM, label="random map, replicate table"), Line2D([], [], marker="D", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK, label="hand write, one matrix")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "update_blocks.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    read1_shares()
    read1_gaps()
    read2_shares()
    paired()
    update_blocks()
