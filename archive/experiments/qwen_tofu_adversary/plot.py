"""E89's figures from results/results.json, the per-step lines of results/run.log and the two earlier result files that hold u1's reads; no model, no new measurement.

  figures/ratio_by_seed.png    the adversary's ratio per seed for W0, do-nothing and the rarely-on half, this update beside u1, against the filed bars
  figures/ratio_vs_left.png    the same ratios against the share of the update left in Δ, on both updates
  figures/steps.png            KL[state's clean ‖ state masked] per step, both seeds, with the final read

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_tofu_adversary/plot.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
COLOUR = {"W0": "#57564f", "N": "#7b4fc4", "Rare": "#c8402f"}  # as E85's figures
NAME = {"W0": "W0, unchanged", "N": "Do-nothing", "Rare": "Rarely-on half, remainder in Δ"}
PASS_AT, FAIL_AT = 1.40, 1.61  # filed, results/predictions.json
TICKS = [1, PASS_AT, FAIL_AT, 2, 3, 4, 5, 7]

R = json.loads((RES / "results.json").read_text())
S = R["score"]
U1_N = json.loads((HERE.parent / "qwen_d1_adversary" / "results" / "amend2.json").read_text())["score"]  # do-nothing on u1; its starts are in amend1.json, H200 NVL, run_amend1.log
U1_RARE = json.loads((HERE.parent / "qwen_d1_teeth" / "results" / "results.json").read_text())["score"]  # the rarely-on half on u1, H200 NVL
SPREAD = S["W0_spread"]


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


def ratio_axis(ax: Axes) -> None:
    ax.set_yscale("log")
    ax.set_yticks(TICKS, [f"{t:g}" for t in TICKS])
    ax.minorticks_off()
    ax.set_ylim(0.9, 8.5)


def bars(ax: Axes, x: float, spread_x: float | None, ha: str = "left") -> None:
    """The filed bars, and W0's own spread between two starts, on a ratio axis."""
    ax.axhspan(1.0, 1 + SPREAD, color="#ecebe6", zorder=0)
    ax.axhline(FAIL_AT, lw=1.3, color="#a32a1e")
    ax.axhline(PASS_AT, lw=1.3, color="#1c7d55")
    ax.text(x, FAIL_AT * 1.012, f"fail bar {FAIL_AT:.2f}", color="#a32a1e", fontsize=9, va="bottom", ha=ha)
    ax.text(x, PASS_AT / 1.012, f"pass bar {PASS_AT:.2f}", color="#1c7d55", fontsize=9, va="top", ha=ha)
    if spread_x is not None:
        ax.text(spread_x, 1.01, f"grey: W0's two starts differ by {SPREAD:.1%}", color=MUTED, fontsize=8.5, va="bottom", ha=ha)


def seeds_and_mean(ax: Axes, x: float, state: str, seeds: list[float], mean: float, text: str) -> None:
    c = COLOUR[state]
    ax.scatter([x - 0.07, x + 0.07], seeds, color=c, s=46, zorder=3)
    ax.hlines(mean, x - 0.2, x + 0.2, color=c, lw=2.6, zorder=3)
    ax.text(x + 0.24, mean, text, color=c, va="center", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=4)


def ratio_by_seed() -> None:
    """One claim: do-nothing passes on u1 and fails on this update, with the rarely-on half above it on each."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 5.4), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    for ax, spread_x in ((left, 1.2), (right, None)):
        style(ax)
        ratio_axis(ax)
        bars(ax, -0.45, spread_x)
    seeds_and_mean(left, 0, "W0", [1.0, 1.0], 1.0, "1 by definition")
    for x, st in ((1, "N"), (2, "Rare")):
        seeds_and_mean(left, x, st, S["r"][st], S["mean"][st], f"mean about {S['mean'][st]:.0f}")
    left.set_xticks(range(3), ["W0, unchanged", "Do-nothing", "Rarely-on half,\nremainder in Δ"])
    left.set_title("This update: TOFU, one wording, 9.6 times u1\nH200 NVL, one pod, seeds 0 and 1", color=INK, fontsize=10.5, loc="left")
    left.set_ylabel("KL[clean ‖ masked], over W0's at the same seed\n(dots: seeds; bar: their mean)")
    seeds_and_mean(right, 0, "W0", [1.0, 1.0], 1.0, "1 by definition")
    seeds_and_mean(right, 1, "N", U1_N["r"]["N"], U1_N["mean"]["N"], f"mean {U1_N['mean']['N']:.2f}")
    seeds_and_mean(right, 2, "Rare", U1_RARE["r"]["Rare"], U1_RARE["mean"]["Rare"], f"mean {U1_RARE['mean']['Rare']:.2f}")
    right.set_xticks(range(3), ["W0, unchanged", "Do-nothing\nE80 Amendment 1\nH200 NVL\nseeds 0, 1", "Rarely-on half,\nremainder in Δ\nE85, another pod\nH200 NVL, seeds 0, 1"])
    right.set_title("u1: the first maths fine-tune update, ten steps\nread earlier, same rows and settings", color=INK, fontsize=10.5, loc="left")
    for ax in (left, right):
        ax.set_xlim(-0.5, 2.85)
    fig.suptitle("Does the adversary's ratio cross the fail bar on a larger update?", color=INK, fontsize=12.5, x=0.01, ha="left")
    finish(fig, "ratio_by_seed.png")


def ratio_vs_left() -> None:
    """One claim: the ratio does not track the share of the update in Δ."""
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    style(ax)
    ratio_axis(ax)
    bars(ax, 0.42, 0.42)
    left = {"this": S["left_in_delta"]["Rare"], "u1": U1_RARE["left_in_delta"]["Rare"]}
    ax.scatter([0], [1], color=COLOUR["W0"], s=60, zorder=3)
    ax.text(0.015, 1.03, "W0: nothing changed", color=COLOUR["W0"], fontsize=9.5, va="bottom")
    points = (("N", 1.0, S["mean"]["N"], "o", "Do-nothing, this update"), ("Rare", left["this"], S["mean"]["Rare"], "o", "Rarely-on half, this update"),
              ("N", 1.0, U1_N["mean"]["N"], "D", "Do-nothing, u1"), ("Rare", left["u1"], U1_RARE["mean"]["Rare"], "D", "Rarely-on half, u1"))
    for state, x, y, mark, label in points:
        c = COLOUR[state]
        ax.scatter([x], [y], color=c if mark == "o" else "white", edgecolor=c, marker=mark, s=80, lw=1.6, zorder=3, label=label)
    for state, x, ys, mark in (("N", 1.0, S["r"]["N"], "o"), ("Rare", left["this"], S["r"]["Rare"], "o"), ("N", 1.0, U1_N["r"]["N"], "D"), ("Rare", left["u1"], U1_RARE["r"]["Rare"], "D")):
        ax.vlines(x, min(ys), max(ys), color=COLOUR[state], lw=1.2, alpha=0.6, zorder=2)
    ax.set_xlim(-0.04, 1.08)
    ax.set_xlabel("Share of the update's squared size left in Δ (plain, pooled over the 196 matrices)")
    ax.set_ylabel("Mean ratio; line: the two seeds")
    ax.set_title("Does the ratio follow the share of the update left in Δ?", color=INK, fontsize=12.5, loc="left")
    ax.legend(frameon=False, loc="upper center", fontsize=9.5, ncol=1, bbox_to_anchor=(0.62, 1.0))
    finish(fig, "ratio_vs_left.png")


def traces() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in (RES / "run.log").read_text().splitlines():
        m = re.search(r"\[(\w+_W_s\d)\] seed \d step (\d+): KL ([\d.]+)", line)
        if m:
            out.setdefault(m[1], []).append(float(m[3]))
    return out


def steps() -> None:
    """One claim: both states are over W0 before the first step, and every read is still climbing at the last."""
    tr = traces()
    assert all(len(v) == 20 for v in tr.values()) and len(tr) == 6
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharey=True)
    for ax, seed in zip(axes, (0, 1), strict=True):
        style(ax, "both")
        for state in ("W0", "N", "Rare"):
            key = f"{state}_W_s{seed}"
            c = COLOUR[state]
            ax.plot(range(20), tr[key], color=c, lw=1.8, label=f"{NAME[state]}, each step")
            ax.scatter([20], [R["starts"][key]["kl"]], color=c, s=55, marker="*", zorder=3, label=f"{NAME[state]}, final read")
        ax.set_yscale("log")
        ax.set_title(f"Seed {seed}", color=INK, fontsize=10.5, loc="left")
        ax.set_xticks([0, 5, 10, 15, 20])
    fig.supxlabel("Adversary step: each line's read comes before that step's move, and the star is the final read, after the last move", fontsize=10)
    axes[0].set_ylabel("KL[state's clean ‖ state masked], nats per token")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Had the search finished at step 20, and when do the states leave W0?", color=INK, fontsize=12.5, x=0.01, ha="left")
    finish(fig, "steps.png")


if __name__ == "__main__":
    ratio_by_seed()
    ratio_vs_left()
    steps()
