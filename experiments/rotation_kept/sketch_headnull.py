"""The planned figure of P1.60 run 2, drawn with made-up numbers before the run (one panel per outcome).

    ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/rotation_kept/sketch_headnull.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
GROUPS = ["copy fine-tune", "lookup fine-tune"]
FT = [[0.131, 0.095, 0.141], [0.018, 0.020, 0.015]]  # E277's saved kept share on all of D
MADE_UP = {  # each arm's three matched randoms, made up
    "randoms near chance: copy above it": [(0.010, 0.020), (0.010, 0.020)],
    "randoms near 0.1: the sentence changes": [(0.090, 0.130), (0.090, 0.130)],
}


def main() -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, bands) in zip(axes, MADE_UP.items(), strict=True):
        FG.column_dots(ax, dict(zip(GROUPS, FT, strict=True)), method="neuron", ylabel="head (2,4) kept share (made-up randoms)",
                       bands=dict(zip(GROUPS, bands, strict=True)))  # fmt: skip
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 0.2)
        rows += [{"panel": title, "group": g, "value": v} for g, vs in zip(GROUPS, FT, strict=True) for v in vs]
        rows += [{"panel": title, "group": f"{g} randoms", "value": b} for g, bs in zip(GROUPS, bands, strict=True) for b in bs]
    axes[0].legend(fontsize=7)
    FG.save(fig, HERE / "figures" / "sketch_headnull.png", n="one point per seed (3 per arm); band = that arm's 3 size-matched randoms, made up", data=rows)


if __name__ == "__main__":
    main()
