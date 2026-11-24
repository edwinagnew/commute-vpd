"""The planned figure of P1.59, drawn with made-up numbers before any reading was computed: one panel per rival.

    ~/commute-agents/fleet/cpu-run python3 experiments/why_fail/sketch.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
GROUPS = ["x2, x10", "random", "toy trained", "procedural", "RL", "realistic"]
MADE_UP = {  # median kept share of the top 1% per item, a few items per class, made up
    "A: the most-needed are rotated": [[0.95, 0.93], [0.01, 0.02, 0.01], [0.05, 0.08, 0.04, 0.10], [0.06, 0.03, 0.07], [0.04, 0.09], [0.06, 0.05, 0.09]],
    "B: a rescaled core, diluted": [[0.95, 0.93], [0.01, 0.02, 0.01], [0.70, 0.62, 0.81, 0.55], [0.66, 0.72, 0.58], [0.61, 0.75], [0.64, 0.70, 0.59]],
}


def main() -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, vals) in zip(axes, MADE_UP.items(), strict=True):
        FG.column_dots(ax, dict(zip(GROUPS, vals, strict=True)), method="neuron", ylabel="kept share, top 1% most-needed (item median)")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=8)
        rows += [{"panel": title, "group": g, "value": v} for g, vs in zip(GROUPS, vals, strict=True) for v in vs]
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; one point per item (fine-tune, built edit or random update)", data=rows)


if __name__ == "__main__":
    main()
