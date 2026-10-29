"""Figure for E50 (restore_anchors), read from results/restore_anchors.json and results/predictions.json.
Plan: reports/plans/2026-10-29_plan_restore-anchors.md; analysis: reports/analyses/2026-10-29_analysis_e50-restore-anchors.md.

  shares.png   per fine-tune, the share of the gain each set takes back when restored (every alive, S, gold, D, the
               largest of the 20 random sets), copy beside random map, with the filed expected region of each class
               shaded on the same axes, and the two references: 0 (nothing restored) and 1 (control R2, every
               matrix's whole key-position output set to W0's, which reads exactly T0 on all eight)

  amend1.png   Amendment 1 (results/amend1.json): every alive subcomponent and the write-only restore on the
               fine-tunes and on the hand writes, the low-reading control

Usage (repo root): ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/restore_anchors/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "restore_anchors.json").read_text())
P = json.loads((HERE / "results" / "predictions.json").read_text())
CLS = P["states"]["classes"]
COPY_C, RAND_C, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
SETS = ["every_alive", "S", "gold", "D", "random_max"]
NAMES = {"every_alive": "every alive\nsubcomponent", "S": "S: Γ's\nswitched-on set", "gold": "gold: W₀'s pieces\nbefore the answers",
         "D": "D: ranked by\npull on target", "random_max": "random sets\n(largest of 20)"}
MARK = {"ident32": "o", "ident32b": "s", "rand32": "o", "rand32b": "s"}


def filed(cls: str, X: str) -> tuple[float, float] | None:
    """The filed expected region (lo, hi) of a class on set X; None where nothing was filed as a band."""
    e, r = P["expected"][cls], P["expected"]["random_sets"]
    if X == "random_max":
        return (-0.05, r["max_max"])
    if cls == "copy":
        return {"every_alive": (e["every_alive_min"], 1.05), "S": (e["S_min"], 1.05), "gold": (e["gold_min"], 1.05)}.get(X)
    return {"every_alive": (-0.05, e["every_alive_max"]), "S": (-0.05, e["S_max"]), "gold": (-0.05, e["gold_max"]),
            "D": (-0.05, e["D_max"])}[X]


def value(st: dict, X: str) -> float:
    return st["random_max"] if X == "random_max" else st["rho"][X]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, X in enumerate(SETS):
        for j, (cls, col) in enumerate((("copy", COPY_C), ("random", RAND_C))):
            x = i + (j - 0.5) * 0.38
            band = filed(cls, X)
            if band:
                ax.add_patch(Rectangle((x - 0.16, band[0]), 0.32, band[1] - band[0], color=col, alpha=0.13, lw=0))
            for key, st in R["states"].items():
                table, seed = key.split("/")
                if CLS[table] != cls:
                    continue
                dx = (0.06 if seed == "s1" else -0.06)
                ax.scatter(x + dx, value(st, X), marker=MARK[table], s=38, color=col, edgecolor=INK, lw=0.5, zorder=3)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.text(len(SETS) - 0.55, 1.015, "whole key-position output set to W₀'s (control R2): 1 on all eight", ha="right",
            va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0.0, color=MUTED, lw=1)
    ax.text(-0.55, -0.015, "nothing taken back (the W₀ control reads 0 for every set)", ha="left", va="top",
            fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(SETS)), [NAMES[X] for X in SETS], fontsize=9)
    ax.set_xlim(-0.6, len(SETS) - 0.4)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("share of the fine-tune's gain taken back\nwhen the set is restored at the key position")
    ax.set_title("Does restoring W₀'s subcomponents take back more of a copy fine-tune's gain than a random map's?\n"
                 "No: the two scored sets, every alive and S, read the two kinds alike, outside the filed regions", fontsize=10.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COPY_C, mec=INK, label="copy fine-tune (existing kind)"),
               plt.Line2D([], [], marker="o", ls="", color=RAND_C, mec=INK, label="random-map fine-tune (new kind)"),
               plt.Line2D([], [], marker="o", ls="", color="white", mec=INK, label="circle table A, square table B;\nleft dot seed 0, right seed 1"),
               Patch(color=COPY_C, alpha=0.25, label="filed region, copy"),
               Patch(color=RAND_C, alpha=0.25, label="filed region, random map")]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.6, 0.9), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "shares.png", dpi=150)



def amend1() -> None:
    """amend1.png: share of the gain taken back by every alive subcomponent and by the write-only restore, for the
    copy and random-map fine-tunes and the four hand writes (the low-reading control), with the power bar."""
    A = json.loads((HERE / "results" / "amend1.json").read_text())
    groups = [("copy\nfine-tunes", COPY_C, lambda k: CLS[k.split("/")[0]] == "copy" and k.split("/")[1] in ("s0", "s1")),
              ("random-map\nfine-tunes", RAND_C, lambda k: CLS[k.split("/")[0]] == "random" and k.split("/")[1] in ("s0", "s1")),
              ("hand writes\n(no learned machinery)", "#2e8b57", lambda k: k.endswith("/write"))]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (_, col, keep) in enumerate(groups):
        rows = [(k, r) for k, r in A["rows"].items() if keep(k)]
        for j, (k, r) in enumerate(rows):
            dx = (j - (len(rows) - 1) / 2) * 0.07
            ax.scatter(i - 0.15 + dx, r["rho"]["every_alive"], marker=MARK[k.split("/")[0]], s=40, color=col, edgecolor=INK, lw=0.5, zorder=3)
            ax.scatter(i + 0.15 + dx, r["rho"]["write_only"], marker=MARK[k.split("/")[0]], s=40, color="white", edgecolor=col, lw=1.2, zorder=3)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.text(2.45, 1.015, "whole key-position output set to W₀'s: 1", ha="right", va="bottom", fontsize=8.5)
    ax.axhline(0.0, color=MUTED, lw=1)
    ax.axhline(0.2, color="#2e8b57", lw=1, ls=":")
    ax.text(2.45, 0.215, "filed ceiling for the hand writes, 0.2", ha="right", va="bottom", fontsize=8.5, color="#2e8b57")
    ax.set_xticks(range(3), [g[0] for g in groups], fontsize=9)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.45, 1.1)
    ax.set_ylabel("share of the gain taken back at the key position")
    ax.set_title("Can the every-alive share read low? Yes: direct hand writes read below 0,\n"
                 "while both kinds of fine-tune read about 0.7", fontsize=10.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=MUTED, mec=INK, label="every alive subcomponent restored"),
               plt.Line2D([], [], marker="o", ls="", color="white", mec=INK, label="write-only restore: W₀'s output plus the update's own write"),
               plt.Line2D([], [], marker="o", ls="", color="white", mec="white", label="circle table A, square table B")]
    ax.legend(handles=handles, fontsize=8, loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "amend1.png", dpi=150)


if __name__ == "__main__":
    main()
    amend1()
