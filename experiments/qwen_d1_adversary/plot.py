"""E80's figures from results/results.json and the per-step lines of results/run.log; no model.

  figures/adversary_maths.png  adversary KL on the maths rows: W0's two starts, do-nothing, in-span, A, with the filed bars
  figures/weight_error.png     adversary ratio to W0 against the weight error, with the filed bars
  figures/steps.png            adversary KL per step, on maths and FineWeb rows, with the earlier job's window

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_d1_adversary/plot.py
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
COLOUR = {"W0": "#57564f", "W0 start 2": "#a3a29b", "do-nothing": "#7b4fc4", "in-span": "#e07b00", "A": "#1baf7a"}
MARK = {"W0": "P", "W0 start 2": "P", "do-nothing": "*", "in-span": "D", "A": "s"}
RESULTS = json.loads((RES / "results.json").read_text())
KL = {k: v["kl"] for k, v in RESULTS["starts"].items()}
W0 = KL["W0_M_s0"]
FILED_DO_NOTHING = 1 + RESULTS["score"]["b"]
PASS_AT, FAIL_AT = 1.40, 1.61  # the filed in-span bars, results/predictions.json
EARLIER_W, EARLIER_LOW, EARLIER_HIGH = 1.7016, 1.36, 2.04  # the 128-row job and the filed window, results/predictions.json


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


def adversary_maths() -> None:
    rows = [("W0", "W0, unchanged\nseed 0", "W0_M_s0"), ("W0 start 2", "W0, unchanged\nseed 1", "W0_M_s1"),
            ("do-nothing", "Do-nothing\nupdate in Δ", "N_M_s0"), ("in-span", "In-span\nleast-squares change", "I_M_s0"),
            ("A", "A\nclean-term change", "A_M_s0")]
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    for x, (arm, _, key) in enumerate(rows):
        ax.bar(x, KL[key], color=COLOUR[arm], width=0.62)
        note = "\nthe reference" if key == "W0_M_s0" else f"\n{KL[key] / W0:.3f} x seed 0"
        ax.text(x, 0.25, f"{KL[key]:.2f}{note}", ha="center", va="bottom", color="white", fontsize=10)
    bars = [(FILED_DO_NOTHING, "do-nothing fails above 1.19 x W0 seed 0", "--"),
            (PASS_AT, "in-span and A pass at or under 1.40 x W0 seed 0", "-"),
            (FAIL_AT, "in-span and A fail at or over 1.61 x W0 seed 0", ":")]
    for ratio, label, ls in bars:
        ax.axhline(ratio * W0, color=INK, ls=ls, lw=1.1)
        ax.text(-0.42, ratio * W0 + 0.06, label, color=INK, fontsize=9, va="bottom")
    ax.set_xticks(range(len(rows)), [r[1] for r in rows], color=INK)
    ax.set_xlim(-0.48, 4.48)
    ax.set_ylim(0, 7.1)
    ax.set_ylabel("adversary's final KL, nats per token,\n64 maths rows")
    ax.set_title("Do the filed bars separate do-nothing from the placements? Do-nothing itself sits under the pass bar", color=INK, fontsize=11)
    style(ax)
    finish(fig, "adversary_maths.png")


def weight_error() -> None:
    w = RESULTS["weights"]
    r = RESULTS["score"]["r"]
    points = [("W0", "W0, seed 0", 1.0, 1.0, (-8, -4), "right"), ("W0 start 2", "W0, seed 1", 1.0, KL["W0_M_s1"] / W0, (-8, -4), "right"),
              ("do-nothing", "do-nothing", w["N"], r["N"], (-9, 3), "right"), ("in-span", "in-span", w["I"], r["I"], (9, -14), "left"),
              ("A", "A", w["A"], r["A"], (9, -4), "left")]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for arm, label, x, y, offset, ha in points:
        ax.scatter(x, y, color=COLOUR[arm], marker=MARK[arm], s=130 if arm == "do-nothing" else 70, zorder=3)
        ax.annotate(f"{label}: {y:.3f}", (x, y), textcoords="offset points", xytext=offset, ha=ha, color=INK, fontsize=10)
    for ratio, label, ls in ((FILED_DO_NOTHING, "do-nothing's fail bar, 1.19", "--"), (PASS_AT, "pass bar for in-span and A, 1.40", "-")):
        ax.axhline(ratio, color=INK, ls=ls, lw=1.1)
        ax.text(1.62, ratio + 0.008, label, color=INK, fontsize=9, ha="right", va="bottom")
    ax.set_xlim(0, 1.62)
    ax.set_ylim(0.8, 1.5)
    ax.set_xlabel("weight error: squared distance of what the components hold from the fine-tuned weights,\nover the update's squared size. 0 holds the whole update; 1 holds none of it")
    ax.set_ylabel("adversary KL / W0's seed 0, maths rows")
    ax.set_title("Does the adversary's read follow how much of the update the components hold? Not visibly", color=INK, fontsize=11)
    style(ax, "both")
    finish(fig, "weight_error.png")


def steps_of(tag: str) -> list[float]:
    """The start's KL before each of its 20 sign steps, from the log, then its final KL."""
    pattern = re.compile(rf"\[{tag}\] seed \d+ step \d+: KL ([\d.]+)")
    return [float(x) for x in pattern.findall((RES / "run.log").read_text())] + [KL[tag]]


def steps() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True, sharex=True)
    panels = ((left, "64 maths rows, 256 tokens", (("W0", "W0, seed 0", "W0_M_s0"), ("W0 start 2", "W0, seed 1", "W0_M_s1"), ("do-nothing", "do-nothing", "N_M_s0"),
                                                   ("in-span", "in-span", "I_M_s0"), ("A", "A", "A_M_s0"))),
              (right, "128 FineWeb rows, 512 tokens", (("W0", "W0, seed 0", "W0_W_s0"), ("do-nothing", "do-nothing", "N_W_s0"))))
    for ax, title, arms in panels:
        for arm, label, tag in arms:
            ax.plot(steps_of(tag), color=COLOUR[arm], lw=2, marker=MARK[arm], markevery=[20], ms=8, label=label)
            ax.text(20.5, KL[tag], f"{label} {KL[tag]:.2f}", color=INK, fontsize=9, va="center")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("adversary step; 20 is the final, filed read")
        ax.set_xlim(0, 29)
        ax.set_xticks([0, 5, 10, 15, 20])
        style(ax)
    right.axhspan(EARLIER_LOW, EARLIER_HIGH, xmin=0.6, xmax=0.69, color=COLOUR["W0 start 2"], alpha=0.35, lw=0)
    right.text(17.2, EARLIER_HIGH + 0.75, f"W0's filed window, {EARLIER_LOW} to {EARLIER_HIGH},\naround the earlier job's {EARLIER_W:.2f}", color=INK, fontsize=9, ha="center")
    left.set_ylabel("adversary KL, nats per token")
    fig.suptitle("Had the search finished when it stopped? No, and one read moves by tenths of a nat between steps", color=INK, fontsize=11)
    finish(fig, "steps.png")


if __name__ == "__main__":
    adversary_maths()
    weight_error()
    steps()
