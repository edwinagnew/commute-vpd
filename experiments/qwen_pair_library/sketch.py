"""Run 1's planned capacity figure with made-up numbers, one panel per rival's outcome (card P3.5-run1).

~/commute-agents/fleet/cpu-run python experiments/qwen_pair_library/sketch.py
"""

from __future__ import annotations

from pathlib import Path

from commute import figures

OUT = Path(__file__).parent / "figures" / "sketch_run1.png"
PAIRS = ["copy / cipher", "context / lookup", "known / invented facts"]
SIDES = {"reuse": ("#0072B2", "o"), "new": ("#D55E00", "s")}
# Made-up exact match per (pair, side), one dict per rival's outcome.
OUTCOMES = {
    "rival a (made up): every pair fit": {
        ("copy / cipher", "reuse"): 0.98, ("copy / cipher", "new"): 0.01,
        ("context / lookup", "reuse"): 0.96, ("context / lookup", "new"): 0.02,
        ("known / invented facts", "reuse"): 0.97, ("known / invented facts", "new"): 0.015,
    },
    "rival b (made up): a pair misses": {
        ("copy / cipher", "reuse"): 0.62, ("copy / cipher", "new"): 0.01,
        ("context / lookup", "reuse"): 0.95, ("context / lookup", "new"): 0.21,
        ("known / invented facts", "reuse"): 0.97, ("known / invented facts", "new"): 0.015,
    },
}


def main() -> None:
    fig, axes = figures.figure(figures.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, values) in zip(axes, OUTCOMES.items(), strict=True):
        for side, (color, marker) in SIDES.items():
            xs = [values[(p, side)] for p in PAIRS]
            ax.scatter(xs, range(len(PAIRS)), color=color, marker=marker, s=30, label=f"{side} side")
            rows += [{"panel": title, "pair": p, "side": side, "exact_match": x} for p, x in zip(PAIRS, xs, strict=True)]
        figures.reference(ax, 0.90, "reuse bar 0.90")
        figures.reference(ax, 0.05, "new bar 0.05")
        ax.set_yticks(range(len(PAIRS)), PAIRS)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("greedy exact match, 256 held-out keys")
        ax.set_title(title, fontsize=9, pad=14)
        ax.legend(loc="center", fontsize=7, frameon=False)
    fig.suptitle("SKETCH, made-up numbers: a pair is fit when reuse ≥ 0.90 and new ≤ 0.05")
    figures.save(fig, OUT, n="made-up numbers; the run reads 256 held-out keys per side, same keys both sides",
                 data=rows)


if __name__ == "__main__":
    main()
