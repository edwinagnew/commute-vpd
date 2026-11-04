"""E86's figures from results/score_<update>.json; no model.

  figures/kept_vs_t.png      behaviour kept along W0 + t δW for the three updates, with the do-nothing reference, the filed bars and the line a proportional gain would follow
  figures/clean_table.png    K over the ruler for each state on the three updates, with W0's 1 and in-span's bar
  figures/left_in_delta.png  least squares' plain share left in Δ against the update's squared size, with do-nothing's 1

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_d1_fractions/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
UPDATES = (("u1", "First fine-tune, 10 steps", "#2a78c4"), ("u2", "Gold fine-tune, 100 steps", "#e07b00"), ("tofu", "TOFU fine-tune, 1250 steps", "#1c7d55"))
SCORE = {u: json.loads((RES / f"score_{u}.json").read_text()) for u, _, _ in UPDATES}
SATURATED_AT, GRADED_AT, INSPAN_BAR = 0.9, 0.5, 0.02  # filed, results/predictions.json
RARE_LEFT_U1 = 0.2327374950721992  # E85's rarely-on placement on u1, ../qwen_d1_teeth/results/results.json


def style(ax: Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)  # pyright: ignore[reportArgumentType]
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor="white", dpi=130)
    plt.close(fig)


def kept_vs_t() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls=":")
    ax.text(0.72, 0.655, "gain in proportion to t", color=MUTED, fontsize=9, rotation=24)
    ax.axhline(1.0, color="#7b4fc4", lw=1.2)
    ax.text(0.62, 1.05, "fine-tuned model and do-nothing: 1", color="#7b4fc4", fontsize=9, va="bottom")
    ax.axhline(0.0, color="#57564f", lw=1.2)
    ax.text(0.62, 0.012, "W0: 0", color="#57564f", fontsize=9, va="bottom")
    ax.axvline(0.1, color=GRID, lw=1.0)
    ax.plot([0.07, 0.13], [SATURATED_AT, SATURATED_AT], color="#a32a1e", lw=2.2)
    ax.text(0.135, SATURATED_AT - 0.012, "at t = 0.1, 0.9 or over: saturated", color="#a32a1e", fontsize=9, va="top")
    ax.plot([0.07, 0.13], [GRADED_AT, GRADED_AT], color="#1c7d55", lw=2.2)
    ax.text(0.28, GRADED_AT, "at t = 0.1, 0.5 or under: graded", color="#1c7d55", fontsize=9, va="center")
    for u, label, colour in UPDATES:
        k = SCORE[u]["saturation"]["kept"]
        xs = [0.0, 0.1, 0.25, 0.5, 1.0]
        ys = [0.0, k["0.1"], k["0.25"], k["0.5"], k["1.0"]]
        ax.plot(xs, ys, marker="o", color=colour, lw=2, label=f"{label}: {SCORE[u]['saturation']['band']}, {k['0.1']:.2f} at a tenth")
    ax.set_xlabel("t, the share of the update applied to W0, no masks")
    ax.set_ylabel("Behaviour kept: share of the update's loss drop")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    ax.set_title("Does the gain build along the update? On TOFU it does; on the first fine-tune it does not", color=INK, loc="left", fontsize=11)
    style(ax)
    finish(fig, "kept_vs_t.png")


def clean_table() -> None:
    states = [("inspan", "In-span part"), ("upper", "Often-on half"), ("lower", "Rarely-on half"), ("adjoint", "Adjoint split"), ("remainder", "Remainder"), ("random", "Random change"), ("signs", "Sign-flipped change")]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    offset = {"u1": 0.2, "u2": 0.0, "tofu": -0.2}
    for u, label, colour in UPDATES:
        over = SCORE[u]["table"]["over_ruler"]
        ax.scatter([over[k] for k, _ in states], [len(states) - 1 - i + offset[u] for i in range(len(states))], color=colour, s=44, label=label, zorder=3)
    ax.axvline(1.0, color="#57564f", lw=1.2)
    ax.text(1.03, len(states) - 0.55, "W0: 1", color="#57564f", fontsize=9, va="center")
    ax.axvline(INSPAN_BAR, color="#a32a1e", lw=1.2, ls="--")
    ax.text(INSPAN_BAR * 1.05, len(states) - 0.55, "in-span's bar 0.02", color="#a32a1e", fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.004, 2.4)
    ax.set_ylim(-0.5, len(states) - 0.3)
    ax.set_yticks(range(len(states)), [s for _, s in reversed(states)])
    ax.set_xlabel("K = KL(updated model ‖ state) over W0's K, log scale. Updated model and do-nothing read 0")
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(0.72, 0.0))
    ax.set_title("Does the clean table pass good splits and fail wrong changes? Yes, on all three updates", color=INK, loc="left", fontsize=11)
    style(ax, "x")
    finish(fig, "clean_table.png")


def left_in_delta() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for x, (u, label, colour) in enumerate(UPDATES):
        left = SCORE[u]["left_in_delta"]
        ax.bar(x, left["ls"], color=colour, width=0.55)
        ax.text(x, left["ls"] + 0.015, f"{left['ls']:.3f}", ha="center", color=INK, fontsize=10)
    ax.axhline(1.0, color="#7b4fc4", lw=1.2)
    ax.text(2.45, 0.985, "do-nothing: 1, by construction", color="#7b4fc4", fontsize=9, va="top", ha="right")
    ax.axhline(RARE_LEFT_U1, color="#c8402f", lw=1.2, ls="--")
    ax.text(2.45, RARE_LEFT_U1 + 0.012, "rarely-on placement on the first fine-tune: 0.233", color="#c8402f", fontsize=9, va="bottom", ha="right")
    ax.set_xticks(range(3), [f"{label}\nsquared size {SCORE[u]['left_in_delta']['size']:.1f}" for u, label, _ in UPDATES], fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Left in Δ, plain share of squared size")
    ax.set_title("Does least squares leave more in Δ on a larger update? Hardly", color=INK, loc="left", fontsize=11)
    style(ax)
    finish(fig, "left_in_delta.png")


if __name__ == "__main__":
    kept_vs_t()
    clean_table()
    left_in_delta()
