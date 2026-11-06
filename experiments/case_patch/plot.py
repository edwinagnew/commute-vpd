"""E113's figures from results/run.json and predictions.json, no model. Colours as E112's plot.py: real task blue,
twin orange, the original model and the random-direction control grey; seed 1 hatched.

  figures/readout.png  the original model's block-2 case readout at the trigger, held-out windows: share read as the
                       flipped case, unpatched, under the patch, and under each of the 5 random-direction patches,
                       with the 0.70 bar
  figures/flips.png    per state, the share of train-word windows whose label flips under the
                       patch, under the 2x patch (no rule) and under the worst random draw, with the 0.50, 0.20 and
                       0.10 bars

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/case_patch/plot.py
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

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"
STATES = [("real_s0", "real, seed 0", BLUE, ""), ("real_s1", "real, seed 1", BLUE, "//"),
          ("twin_s0", "twin, seed 0", ORANGE, ""), ("twin_s1", "twin, seed 1", ORANGE, "//")]  # fmt: skip


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def label(ax: Any, x: float, v: float, fmt: str = "{:.3f}") -> None:
    ax.text(x, v + 0.01, fmt.format(v), ha="center", va="bottom", fontsize=7)


def readout(r: dict[str, Any], bar: float) -> None:
    Q = r["Q0"]
    vals = [Q["unpatched"]["flipped"], Q["patched"]["flipped"]] + [d["flipped"] for d in Q["random"]]
    xs = [0, 1] + [2.4 + 0.3 * k for k in range(len(Q["random"]))]
    cols = [GREY, BLUE] + [GREY] * len(Q["random"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for x, v, c in zip(xs, vals, cols, strict=True):
        ax.bar(x, v, 0.6 if x < 2 else 0.26, color=c, alpha=1.0 if x < 2 else 0.6)
        label(ax, x, v)
    ax.axhline(bar, color=BLUE, ls="--", lw=1)
    ax.text(3.9, bar + 0.01, f"bar {bar:.2f}", ha="right", va="bottom", color=BLUE, fontsize=8)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(3.9, 0.51, "chance 0.50", ha="right", va="bottom", color=GREY, fontsize=8)
    ax.set(xticks=[0, 1, 3.0], xlim=(-0.5, 4.0), ylim=(0, 1), ylabel="share of windows read as the flipped case",
           xticklabels=["no patch", "case patch\n(block 1, trigger and word)", "5 random-direction patches\n(same step size)"],
           title="Does the patch at block 1 flip the case that block 2 reads?")  # fmt: skip
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "readout.png", dpi=150)


def flips(r: dict[str, Any], rules: dict[str, Any]) -> None:
    B, S, Cn = r["beside"], r["states"], r["control"]
    w = 0.18
    groups = [[S[s]["Q1"] for s, *_ in STATES], [B[s]["flip_2x"] for s, *_ in STATES], [max(Cn[s]) for s, *_ in STATES]]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for x0, vals in enumerate(groups):
        for i, ((_, _, c, h), v) in enumerate(zip(STATES, vals, strict=True)):
            x = x0 + (i - 1.5) * w
            ax.bar(x, v, w, color=GREY if x0 == 2 else c, hatch=h, edgecolor="white")
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.axhline(rules["Q1"]["real"], color=BLUE, ls="--", lw=1.2)
    ax.axhline(rules["Q1"]["twin_expected_max"], color=ORANGE, ls=":", lw=1.2)
    ax.hlines(rules["Q1"]["control_max"], 2 - 2.4 * w, 2 + 2.4 * w, color=GREY, ls="--", lw=1.2)
    ax.set(xticks=[0, 1, 2], xlim=(-0.6, 2.6), ylim=(0, 0.85), ylabel="share of train-word windows whose label flips",
           xticklabels=["case patch\n(the filed test)", "case patch, twice the shift\n(beside, no rule)",
                        "random-direction patch,\nworst of 5 draws"],
           title="Does the fine-tuned label follow the patched case readout?")  # fmt: skip
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=n) for _, n, c, h in STATES]
    handles += [Line2D([], [], color=BLUE, ls="--", label="bar for 'reuses' (0.50)"),
                Line2D([], [], color=ORANGE, ls=":", label="twin expected at most (0.20)"),
                Line2D([], [], color=GREY, ls="--", label="random patch allowed at most (0.10)")]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "flips.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rules, r = load("predictions.json")["rules"], load("run.json")
    readout(r, rules["Q0"]["min"])
    flips(r, rules)


if __name__ == "__main__":
    main()
