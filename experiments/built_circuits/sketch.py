"""The planned figure of P1.63 (Figure 2's built edits on more circuits), made-up numbers, one row of panels per rival.

    ~/commute-agents/fleet/cpu-run python experiments/built_circuits/sketch.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
FEATS = ("VPD subcomponents", "top neurons", "SAE latents")
EDITS = ("x2", "x10", "rebuilt", "new_few", "new_thin")
CIRCUITS = ("Fig 2 (copy24, layer 1)", "induction, layer 2", "rank-1 lookup, layer 0")
# made-up: per rival, featurization, edit, one value per circuit; the matched random of each edit as (low, high)
MADE_UP = {
    "A: the pattern holds on every circuit": {
        "VPD subcomponents": ([0.76, 0.70, 0.66], [0.76, 0.69, 0.64], [0.62, 0.55, 0.50], [0.01, 0.02, 0.01], [0.02, 0.02, 0.03]),
        "top neurons": ([1.0, 0.99, 0.98], [1.0, 0.99, 0.98], [0.05, 0.10, 0.08], [0.00, 0.01, 0.01], [0.01, 0.01, 0.02]),
        "SAE latents": ([0.97, 0.90, 0.85], [0.95, 0.88, 0.84], [0.13, 0.20, 0.15], [0.00, 0.01, 0.01], [0.01, 0.02, 0.02]),
    },
    "B: it is specific to Fig 2's site": {
        "VPD subcomponents": ([0.76, 0.08, 0.15], [0.76, 0.07, 0.12], [0.62, 0.04, 0.05], [0.01, 0.02, 0.01], [0.02, 0.02, 0.03]),
        "top neurons": ([1.0, 0.30, 0.45], [1.0, 0.28, 0.40], [0.05, 0.02, 0.03], [0.00, 0.01, 0.01], [0.01, 0.01, 0.02]),
        "SAE latents": ([0.97, 0.10, 0.20], [0.95, 0.09, 0.18], [0.13, 0.02, 0.03], [0.00, 0.01, 0.01], [0.01, 0.02, 0.02]),
    },
}
RANDOM_BAND = (0.0, 0.03)


def main() -> None:
    fig, axes = FG.figure((10.0, 6.0), ncols=3, nrows=2)
    rows = []
    for r, (rival, by_feat) in enumerate(MADE_UP.items()):
        for c, feat in enumerate(FEATS):
            ax = axes[r][c]
            groups = dict(zip(EDITS, by_feat[feat], strict=True))
            FG.column_dots(ax, groups, method="eb", ylabel="write share over the needed set",
                           bands=dict.fromkeys(EDITS, RANDOM_BAND))
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{feat}\nrival {rival[:1]} (made-up)", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            rows += [{"rival": rival, "feat": feat, "edit": e, "circuit": k, "value": v}
                     for e, vs in groups.items() for k, v in zip(CIRCUITS, vs, strict=True)]
    axes[0][0].legend(fontsize=6, loc="center right")
    fig.suptitle("Sketch: each built edit's write share on three circuits (points), its matched random (band)", fontsize=9)
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; 3 circuits per edit (Fig 2's and two new), "
            "one matched random per edit", data=rows)


if __name__ == "__main__":
    main()
