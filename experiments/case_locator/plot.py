"""E114's figures, one per claim, from results/*.json; no model.

  figures/removal.png  the share of the label-gap gain lost when each of the 28 parts' output is set to W0's mean,
                       real and twin on the same axes, with the located rule's two bars and the two planted edits
  figures/reverts.png  the sum of the 28 single-part reverts per state (unit lo), with E100's 0.40 and 0.62 bars, E100's
                       copy and key-table rows as reference, and the planted edit (written by construction)

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/case_locator/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
E100 = HERE.parent / "fed_or_written" / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


S0, S1, PRED = load(RES / "stage0.json"), load(RES / "stage1.json"), load(RES / "predictions.json")
E100_TABLE, E100_AM1, E100_PRED = load(E100 / "table.json"), load(E100 / "amend1.json"), load(E100 / "predictions_amend1.json")
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8f8e88"
INK, MUTED, GRID, SURFACE = "#2b2a28", "#52514e", "#e4e3df", "#fcfcfb"
STATES = {"real_s0": ("real, seed 0", BLUE, "o"), "real_s1": ("real, seed 1", BLUE, "s"),
          "twin_s0": ("twin, seed 0", ORANGE, "o"), "twin_s1": ("twin, seed 1", ORANGE, "s")}  # fmt: skip
A, CONTROL_MAX = float(PRED["thresholds"]["a"]), float(PRED["thresholds"]["control_max"])
FED_BAR, WRITTEN_BAR = float(E100_PRED["thresholds"]["fed_bar"]), float(E100_PRED["thresholds"]["written_bar"])

# E100 reference rows: task -> (words, construction group)
E100_ROWS = {
    "T5 copy": ("in-context copy", "copy"),
    "T13 copy-24": ("copy table, 24 keys", "copy"),
    "T4 ident32": ("identity table, 32 keys", "copy"),
    "T5 lookup": ("in-context lookup", "new answers"),
    "T11 cipher": ("cipher table, 24 keys", "new answers"),
    "T6 rand32": ("random table, 32 keys", "new answers"),
    "T8 cipher": ("ciphered text", "new answers"),
}


def axes(ax: Any, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def part_name(p: str) -> str:
    return f"block {p[3]} MLP" if p.startswith("mlp") else f"head ({p})"


def removal() -> None:
    """Removal share per part, unit gap, sorted by the real arm's mean share."""
    parts = [p for p in S1["states"]["real_s0"]["removal"] if "+" not in p]
    share = {k: {p: float(S1["states"][k]["removal"][p]["share"]["gap"]) for p in parts} for k in STATES}
    order = sorted(parts, key=lambda p: (share["real_s0"][p] + share["real_s1"][p]) / 2)
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    for i, (k, (label, colour, marker)) in enumerate(STATES.items()):
        dy = (i - 1.5) * 0.12
        ax.scatter([share[k][p] for p in order], [j + dy for j in range(len(order))], color=colour, marker=marker,
                   s=34, edgecolor=SURFACE, linewidth=0.8, zorder=3, label=label)  # fmt: skip
    plants = {"block3": ("planted edit in block 3's MLP", "x"), "block1": ("planted edit in block 1's MLP", "+")}
    for b, (label, marker) in plants.items():
        rm = S0["plants"][b]["reads"]["removal"]
        ax.scatter([float(rm[p]["share"]["gap"]) for p in order], range(len(order)), color=GREY, marker=marker, s=40,
                   linewidth=1.2, zorder=2, label=label)  # fmt: skip
    ax.axvline(A, color=INK, linestyle="--", linewidth=1)
    ax.text(A + 0.01, len(order) - 0.6, f"located rule's bar for the real arm, {A:.2f}", color=INK, fontsize=8)
    ax.axvline(CONTROL_MAX, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(CONTROL_MAX + 0.01, len(order) - 1.4, f"twin counts as spared under {CONTROL_MAX:.2f}", color=MUTED, fontsize=8)
    ax.axvline(0, color=GRID, linewidth=1.5)
    ax.set_yticks(range(len(order)), [part_name(p) for p in order], fontsize=8)
    ax.set_xlim(-0.65, 1.12)
    axes(ax, "share of the label-gap gain lost when the part's output is set to W₀'s mean on these windows\n"
             "(0: the part is not needed; 1: the whole gain goes with it)")  # fmt: skip
    ax.set_title("Does the real fine-tune need a part that the memorising twin does not?", color=INK, loc="left",
                 fontsize=11, pad=10)  # fmt: skip
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "removal.png")


def e100_rows() -> list[tuple[str, list[float]]]:
    vals: dict[str, list[float]] = {t: [] for t in E100_ROWS}
    for r in E100_TABLE["table"] + list(E100_AM1["states"].values()):
        if r["task"] in vals:
            vals[r["task"]].append(float(r["reverts_sum"]))
    return [(f"E100: {w} ({g})", vals[t]) for t, (w, g) in E100_ROWS.items()]


def reverts() -> None:
    """Sum of single reverts per state, unit lo, against E100's bars and rows."""
    rows = e100_rows()
    ours = [("E114: case label, real", ["real_s0", "real_s1"], BLUE), ("E114: case label, twin", ["twin_s0", "twin_s1"], ORANGE)]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, (_, v) in enumerate(rows):
        ax.scatter(v, [i] * len(v), color=GREY, s=26, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    n = len(rows)
    for j, (_, keys, colour) in enumerate(ours):
        ax.scatter([float(S1["states"][k]["sum_own"]["lo"]) for k in keys], [n + j] * 2, color=colour, s=48,
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)  # fmt: skip
    ax.axvspan(FED_BAR, WRITTEN_BAR, color=GRID, alpha=0.6, zorder=1)
    for x, word, dx, ha in ((FED_BAR, "fed at or under", -0.008, "right"), (WRITTEN_BAR, "written at or over", 0.008, "left")):
        ax.axvline(x, color=INK, linestyle="--", linewidth=1)
        ax.text(x + dx, n + 1.6, f"E100: {word} {x:.2f}", color=INK, fontsize=8, ha=ha)
    ax.set_yticks(range(n + 2), [r[0] for r in rows] + [o[0] for o in ours], fontsize=8)
    ax.set_ylim(-0.6, n + 2.1)
    ax.set_xlim(0, 1.05)
    axes(ax, "sum over the 28 parts of the share of the gain lost when that one part's update is reverted\n"
             "(unit: log-odds of the target label; a planted one-matrix edit reads 1 by construction)")  # fmt: skip
    ax.set_title("Is the case-label gain fed through unchanged parts, or written into the parts themselves?",
                 color=INK, loc="left", fontsize=11, pad=10)  # fmt: skip
    handles = [Line2D([], [], marker="o", ls="", color=GREY, label="E100 fine-tunes, one dot per seed"),
               Line2D([], [], marker="o", ls="", color=BLUE, label="real: label by the previous word's case"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="twin: label by a shuffled list")]  # fmt: skip
    ax.legend(handles=handles, loc="center left", fontsize=8, frameon=False)
    save(fig, "reverts.png")


if __name__ == "__main__":
    removal()
    reverts()
