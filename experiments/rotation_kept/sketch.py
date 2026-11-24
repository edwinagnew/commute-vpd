"""The planned figure of P1.60, drawn with made-up numbers before the run (one panel per rival).

    ~/commute-agents/fleet/cpu-run python experiments/rotation_kept/sketch.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
GROUPS = ["x2/x10", "new", "E62 copy", "E62 lookup", "E229 trained", "panel trained"]
NULL = {g: (0.02, 0.08) for g in GROUPS}  # the matched randoms' rotation share minus its null, made up
MADE_UP = {
    "A: trained updates rotate toward M's needed writes": [[0.9, 0.92], [0.03, 0.05], [0.55, 0.6, 0.5], [0.2, 0.25, 0.22], [0.4, 0.5, 0.35], [0.45, 0.3, 0.5]],
    "B: trained updates write somewhere new": [[0.9, 0.92], [0.03, 0.05], [0.06, 0.04, 0.05], [0.05, 0.03, 0.04], [0.06, 0.05, 0.04], [0.04, 0.06, 0.05]],
}


def main() -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, vals) in zip(axes, MADE_UP.items(), strict=True):
        groups = dict(zip(GROUPS, vals, strict=True))
        FG.column_dots(ax, groups, method="neuron", ylabel="rotation share (made up)", bands=NULL)
        ax.set_title(f"rival {title}", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        rows += [{"rival": title[0], "group": g, "value": v} for g, vs in groups.items() for v in vs]
    axes[0].legend(fontsize=7)
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; one point per state (seeds or tasks), band = matched randoms", data=rows)


if __name__ == "__main__":
    main()
