"""The research log's figure for E89: plot.py's ratio figure with plain tick labels (no amendment or run codes).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_tofu_adversary/plot_log.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from commute.jobs import load_experiment

P = load_experiment("qwen_tofu_adversary", "plot")


def log_ratio() -> None:
    """One claim: doing nothing passes on the first update and fails on the larger one; the rarely-on half fails higher."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 5.4), sharey=True)
    for ax, spread_x in ((left, 1.2), (right, None)):
        P.style(ax)
        P.ratio_axis(ax)
        P.bars(ax, -0.45, spread_x)
    P.seeds_and_mean(left, 0, "W0", [1.0, 1.0], 1.0, "1 by definition")
    for x, st in ((1, "N"), (2, "Rare")):
        P.seeds_and_mean(left, x, st, P.S["r"][st], P.S["mean"][st], f"mean about {P.S['mean'][st]:.0f}")
    names = ["Unchanged model", "Doing nothing", "Rarely-on half,\nremainder in Δ"]
    left.set_xticks(range(3), names)
    left.set_title("Larger update: TOFU, one wording, 9.6 times the first", color=P.INK, fontsize=10.5, loc="left")
    left.set_ylabel("Adversary's KL, over the unchanged model's\n(dots: two seeds; bar: their mean)")
    P.seeds_and_mean(right, 0, "W0", [1.0, 1.0], 1.0, "1 by definition")
    P.seeds_and_mean(right, 1, "N", P.U1_N["r"]["N"], P.U1_N["mean"]["N"], f"mean {P.U1_N['mean']['N']:.2f}")
    P.seeds_and_mean(right, 2, "Rare", P.U1_RARE["r"]["Rare"], P.U1_RARE["mean"]["Rare"], f"mean {P.U1_RARE['mean']['Rare']:.2f}")
    right.set_xticks(range(3), names)
    right.set_title("First update: ten maths steps, read earlier", color=P.INK, fontsize=10.5, loc="left")
    for ax in (left, right):
        ax.set_xlim(-0.5, 2.85)
    fig.suptitle("Doing nothing fails the adversary on the larger update, and passed on the first", color=P.INK, fontsize=12.5, x=0.01, ha="left")
    P.finish(fig, "log_ratio.png")


if __name__ == "__main__":
    log_ratio()
