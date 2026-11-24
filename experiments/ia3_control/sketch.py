"""The planned figure of P1.61, drawn with made-up numbers before the run: one panel per rival, E_b over the needed set
K at the top-activating-neuron featurization (the remainders in, as E277 filed it) for each key-grid task, the full
fine-tune beside the FFN-gain arm and its matched random. The full fine-tune's values are made up near E277's trained
reads (0.02-0.04).

    ~/commute-agents/fleet/cpu-run python experiments/ia3_control/sketch.py
"""

from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
TASKS = ["copy", "rank1", "elicit", "cipher", "prior_cipher"]
FULL = [0.03, 0.03, 0.03, 0.02, 0.03]
RANDOM = [0.01, 0.01, 0.01, 0.01, 0.01]
MADE_UP = {
    "A: a trained gain update reads kept": [0.75, 0.7, 0.72, 0.6, 0.68],
    "B: the instrument reads low on any trained update": [0.06, 0.05, 0.06, 0.04, 0.05],
}


def main() -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    for ax, (title, ffn) in zip(axes, MADE_UP.items(), strict=True):
        FG.column_dots(ax, {t: [v] for t, v in zip(TASKS, FULL, strict=True)}, method="neuron_half", ylabel="E_b over K at neurons")
        FG.column_dots(ax, {t: [v] for t, v in zip(TASKS, ffn, strict=True)}, method="neuron", ylabel="E_b over K at neurons",
                       bands={t: (0.0, r) for t, r in zip(TASKS, RANDOM, strict=True)})  # fmt: skip
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=8)
        rows += [{"rival": title[0], "task": t, "full": f, "ffn_gain": g, "random": r} for t, f, g, r in zip(TASKS, FULL, ffn, RANDOM, strict=True)]
    axes[0].scatter([], [], color=FG.METHODS["neuron_half"].color, marker=FG.METHODS["neuron_half"].marker, label="full fine-tune")
    axes[0].scatter([], [], color=FG.METHODS["neuron"].color, marker=FG.METHODS["neuron"].marker, label="FFN-gain fine-tune")
    axes[0].legend(fontsize=7, loc="center right")
    FG.save(fig, HERE / "figures" / "sketch.png", n="made-up numbers; the run: 3 seeds per task, 512 windows each", data=rows)


if __name__ == "__main__":
    main()
