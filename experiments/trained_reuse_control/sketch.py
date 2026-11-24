"""The planned figure of P1.4 run 2, drawn with made-up numbers before the run: one panel per rival, E_b over the needed
set K at the top-activating-neuron featurization (remainders in, as E277 filed it) for the built edits x2 and new_few
(E277's measured values) and for each seed's imitator of each, beside the matched randoms' range. The imitators' values
are made up; the built ones are E277's.

    ~/commute-agents/fleet/cpu-run python experiments/trained_reuse_control/sketch.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
BUILT = {"x2 built": [0.9998], "new_few built": [0.0047]}
RANDOM = (0.0, 0.013)
MADE_UP = {
    "A: an imitator reads as its target's label": {"x2 imitators": [0.62, 0.55, 0.7], "new_few imitators": [0.02, 0.03, 0.02]},
    "B: every trained update reads low": {"x2 imitators": [0.03, 0.025, 0.035], "new_few imitators": [0.02, 0.03, 0.02]},
}


def main() -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, ft) in zip(axes, MADE_UP.items(), strict=True):
        FG.column_dots(ax, BUILT | {k: [None] for k in ft}, method="neuron_half", ylabel="E_b over K at neurons")
        FG.column_dots(ax, {k: [None] for k in BUILT} | ft, method="neuron", ylabel="E_b over K at neurons",
                       bands={k: RANDOM for k in ft})  # fmt: skip
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
        ax.set_title(f"rival {title}\n(made-up imitator values)", fontsize=8)
        rows += [{"rival": title[0], "group": k, "values": v} for k, v in (BUILT | ft).items()]
    axes[0].scatter([], [], color=FG.METHODS["neuron_half"].color, marker=FG.METHODS["neuron_half"].marker, label="built edit (E277)")
    axes[0].scatter([], [], color=FG.METHODS["neuron"].color, marker=FG.METHODS["neuron"].marker, label="imitator, one per seed")
    axes[0].legend(fontsize=7, loc="upper right")
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up imitator values; the run: 3 seeds per target, x2 on 75 windows x 41 positions, new_few on 512", data=rows)


if __name__ == "__main__":
    main()
