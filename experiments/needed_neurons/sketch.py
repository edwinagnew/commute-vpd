"""The planned figure of P1.56, drawn with made-up numbers before the run (one panel per rival).

    ~/commute-agents/fleet/cpu-run python experiments/needed_neurons/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EDITS = ["x2", "x10", "rebuilt", "new_few", "new_thin", "random_size", "random_kl"]
VPD = [0.761, 0.761, 0.623, 0.013, 0.019, 0.011, 0.011]  # E268, E_b over K at VPD (measured)
MADE_UP = {
    "A: the order holds at neurons": [0.62, 0.58, 0.35, 0.03, 0.04, 0.02, 0.02],
    "B: the order collapses at neurons": [0.09, 0.08, 0.06, 0.04, 0.05, 0.03, 0.03],
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, nb) in zip(axes, MADE_UP.items(), strict=True):
        x = range(len(EDITS))
        ax.plot(x, VPD, "s", color="grey", label="VPD (E268, measured)")
        ax.plot(x, nb, "o", label="top-neuron featurization (made up)")
        ax.axhspan(0, max(nb[-2:]) + 0.2, color="C1", alpha=0.1, label="within 0.2 of the randoms")
        ax.set_xticks(list(x), EDITS, rotation=30)
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=9)
    axes[0].set_ylabel("E_b over the needed set K")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
