"""The planned figure of E283 (P1.57), drawn with made-up numbers before any V1 or V2 was computed: one panel per rival.

    ~/commute-agents/fleet/cpu-run python3 experiments/reliance/sketch.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
GROUPS = ["x2", "x10", "new_few", "new_thin", "random_kl", "E62 copy", "E62 lookup"]
MADE_UP = {  # (V1, V2) per group, made up
    "A: reliance separates reuse from new": ([0.99, 0.99, 0.35, 0.40, 0.30, 0.80, 0.45], [0.97, 0.96, 0.30, 0.45, 0.20, 0.75, 0.40]),
    "B: reliance fails too": ([0.99, 0.99, 0.95, 0.96, 0.97, 0.70, 0.72], [0.97, 0.96, 0.95, 0.90, 0.93, 0.60, 0.62]),
}


def main() -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, (v1, v2)) in zip(axes, MADE_UP.items(), strict=True):
        x = range(len(GROUPS))
        ax.scatter([i - 0.12 for i in x], v1, color=FG.METHODS["neuron"].color, marker="o", s=24, label="V1: needed by both")
        ax.scatter([i + 0.12 for i in x], v2, color=FG.METHODS["steering"].color, marker="D", s=24, label="V2: needed and barely changed")
        ax.set_xticks(list(x), GROUPS, rotation=30, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=8)
        rows += [{"panel": title, "group": g, "V1": a, "V2": b} for g, a, b in zip(GROUPS, v1, v2, strict=True)]
    for ax in axes:
        ax.set_ylabel("share of F's flip mass on its needed set", fontsize=7)
    axes[0].legend(fontsize=7, loc="lower left")
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; one built edit each, three seeds per E62 side", data=rows)


if __name__ == "__main__":
    main()
