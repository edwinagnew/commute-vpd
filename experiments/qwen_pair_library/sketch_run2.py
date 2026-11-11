"""Run 2's planned figure with made-up numbers, one panel per rival's outcome (card P3.5-run2): held-out exact match of
the copy and cipher sides at the 5x and 15x saves.

~/commute-agents/fleet/cpu-run python experiments/qwen_pair_library/sketch_run2.py
"""

from __future__ import annotations

from pathlib import Path

from commute import figures

OUT = Path(__file__).parent / "figures" / "sketch_run2.png"
SIZES = ["5x the floor", "15x the floor"]
SIDES = {"copy (reuse)": ("#0072B2", "o"), "cipher (new)": ("#D55E00", "s")}
# Made-up held-out exact match per (side, size), one dict per rival's outcome.
OUTCOMES = {
    "rival a (made up): only copy generalises": {
        ("copy (reuse)", "5x the floor"): 0.93, ("copy (reuse)", "15x the floor"): 1.0,
        ("cipher (new)", "5x the floor"): 0.02, ("cipher (new)", "15x the floor"): 0.08,
    },
    "rival b (made up): the letter map is a rule": {
        ("copy (reuse)", "5x the floor"): 0.93, ("copy (reuse)", "15x the floor"): 1.0,
        ("cipher (new)", "5x the floor"): 0.71, ("cipher (new)", "15x the floor"): 0.88,
    },
}


def main() -> None:
    fig, axes = figures.figure(figures.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, values) in zip(axes, OUTCOMES.items(), strict=True):
        for side, (color, marker) in SIDES.items():
            xs = [values[(side, s)] for s in SIZES]
            ax.scatter(xs, range(len(SIZES)), color=color, marker=marker, s=30, label=side)
            rows += [{"panel": title, "side": side, "size": s, "held_out_exact_match": x} for s, x in zip(SIZES, xs, strict=True)]
        figures.reference(ax, 0.70, "copy at 15x minus the 0.3 gap")
        ax.set_yticks(range(len(SIZES)), SIZES)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("greedy exact match, 256 held-out keys, one seed")
        ax.set_title(title, fontsize=9, pad=14)
        ax.legend(loc="center", fontsize=7, frameon=False)
    fig.suptitle("SKETCH, made-up numbers: does the cipher side generalise to held-out keys as copy does?")
    figures.save(fig, OUT, n="made-up numbers; the run reads 256 held-out keys per side, same keys both sides, 3 seeds",
                 data=rows)


if __name__ == "__main__":
    main()
