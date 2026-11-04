"""E80 Amendments 1 and 2's two figures, from results/amend2.json and results/amend1.json; no model.

  figures/faithful_seeds.png  adversarial ratio per seed for the four methods and in-span, with the pass and fail bars and W0's spread
  figures/kept_fraction.png   share of the gain kept against fraction of the update, beside the four methods' shares

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_d1_adversary/plot_amend2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID, BAND = "#2b2a27", "#6f6e69", "#e4e3df", "#d9d8d2"
PASS_AT, FAIL_AT = 1.40, 1.61  # the slide's bars, results/predictions_amend2.json
KEPT_GOOD, KEPT_BAD = 0.9, 0.5  # the slide's kept-share bands, as in plot_scorecard.py
NAMES = {"N": "Do-nothing", "LS": "Least squares", "ADJ": "Adjoint split", "A": "Teacher training", "I": "In-span"}
COLOUR = {"N": "#7b4fc4", "I": "#e07b00", "LS": "#1aa1d6", "ADJ": "#b3366f", "A": "#1baf7a"}  # N, I and A as in plot.py
A2 = json.loads((RES / "amend2.json").read_text())["score"]
A1 = json.loads((RES / "amend1.json").read_text())["score"]


def style(ax: Axes, grid: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)  # pyright: ignore[reportArgumentType]
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, facecolor="white", dpi=130, bbox_inches="tight")
    plt.close(fig)


def faithful_seeds() -> None:
    ratios = {k: A2["r"][k] for k in ("N", "LS", "ADJ", "A")} | {"I": A1["r"]["I"]}
    spread = A2["W0_spread"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    style(ax, "x")
    ax.axvspan(1, 1 + spread, color=BAND, lw=0, zorder=0)
    ax.axvline(1, color=MUTED, lw=1.2)
    ax.axvline(PASS_AT, color=INK, lw=1.2, ls="--")
    ax.axvline(FAIL_AT, color=INK, lw=1.2)
    ys = range(len(ratios), 0, -1)
    for y, (k, (s0, s1)) in zip(ys, ratios.items(), strict=True):
        ax.plot([s0, s1], [y, y], color=COLOUR[k], lw=2, zorder=2)
        ax.scatter(s0, y, marker="o", s=70, color=COLOUR[k], edgecolor="white", zorder=3)
        ax.scatter(s1, y, marker="s", s=60, color=COLOUR[k], edgecolor="white", zorder=3)
        ax.scatter((s0 + s1) / 2, y, marker="|", s=170, color=INK, lw=2, zorder=4)
    ax.set_yticks(list(ys), [f"{NAMES[k]}, mean {(s0 + s1) / 2:.2f}" for k, (s0, s1) in ratios.items()], color=INK)
    top = len(ratios) + 0.62
    ax.set(xlim=(0.95, 1.72), ylim=(0.4, top + 0.55))
    ax.text(1 + spread / 2, top, f"unchanged decomposition\nreads 1; its two seeds\ndiffer by {spread:.1%}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(PASS_AT, top, f"pass at or\nunder {PASS_AT:.2f}", ha="center", va="center", fontsize=8.5, color=INK, backgroundcolor="white")
    ax.text(FAIL_AT, top, f"fail at or\nover {FAIL_AT:.2f}", ha="center", va="center", fontsize=8.5, color=INK, backgroundcolor="white")
    ax.set_xlabel("Adversarial ratio on web rows: state's final damage over the unchanged decomposition's, same seed", color=INK)
    ax.set_title("Does the adversary fail doing nothing? No: every state passes, doing nothing included", color=INK, fontsize=11.5, loc="left")
    keys = [Line2D([], [], marker="o", color=MUTED, ls="", ms=8, label="seed 0"), Line2D([], [], marker="s", color=MUTED, ls="", ms=7, label="seed 1"),
            Line2D([], [], marker="|", color=INK, ls="", ms=12, mew=2, label="mean of the two")]
    ax.legend(handles=keys, loc="lower right", frameon=False, fontsize=9)
    finish(fig, "faithful_seeds.png")


def kept_fraction() -> None:
    gold = A2["gold"]
    xs = [0.0, 0.1, 0.25, 1.0]
    ys = [(gold["W0"] - loss) / (gold["W0"] - gold["FT"]) for loss in (gold["W0"], gold["t0.1"], gold["t0.25"], gold["t1.0"])]
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(8.6, 4.0), width_ratios=(1.5, 1), layout="constrained")
    for ax in (left, right):
        style(ax, "y")
        ax.axhline(KEPT_GOOD, color=INK, lw=1.2, ls="--")
        ax.axhline(KEPT_BAD, color=INK, lw=1.2)
        ax.axhline(0, color=MUTED, lw=1.2)
    left.plot(xs, ys, color=MUTED, lw=2, marker="o", ms=8, mec="white", zorder=3)
    left.text(0.12, ys[1] - 0.075, f"a tenth: {ys[1]:.2f}", fontsize=9, color=INK)
    left.text(0.27, ys[2] + 0.04, f"a quarter: {ys[2]:.2f}", fontsize=9, color=INK)
    left.text(1.0, ys[3] + 0.04, f"the whole update: {ys[3]:.2f}", ha="right", fontsize=9, color=INK)
    left.annotate("no update, the unchanged model: 0", (0, ys[0]), xytext=(0.03, 0.03), fontsize=9, color=INK)
    left.text(1.0, KEPT_GOOD - 0.015, f"good at or over {KEPT_GOOD}", ha="right", va="top", fontsize=8.5, color=INK)
    left.text(1.0, KEPT_BAD - 0.015, f"bad under {KEPT_BAD}", ha="right", va="top", fontsize=8.5, color=INK)
    left.set(xlim=(-0.02, 1.03), ylim=(-0.05, 1.15))
    left.set_xlabel("Fraction of the update added to W0", color=INK)
    left.set_ylabel("Share of the held-out gain kept", color=INK)
    left.set_title("Does the kept share need the whole update? No: a tenth clears the good line", color=INK, fontsize=11.5, loc="left")
    methods = ("N", "LS", "ADJ", "A")
    for x, k in enumerate(methods):
        right.scatter(x, A2["kept"][k], s=70, color=COLOUR[k], edgecolor="white", zorder=3)
        right.text(x, A2["kept"][k] + 0.04, f"{A2['kept'][k]:.2f}", ha="center", fontsize=9, color=INK)
    right.set_xticks(range(4), ["Do-\nnothing", "Least\nsquares", "Adjoint\nsplit", "Teacher\ntraining"], color=INK)
    right.set_xlim(-0.5, 3.5)
    right.set_xlabel("The four methods' states", color=INK)
    finish(fig, "kept_fraction.png")


if __name__ == "__main__":
    faithful_seeds()
    kept_fraction()
