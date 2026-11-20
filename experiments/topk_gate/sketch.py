"""The planned figure of P2.47, drawn with made-up numbers before the run (one panel per rival).

    ~/commute-agents/fleet/cpu-run python3 experiments/topk_gate/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PAIRS = ["copy/lookup", "ident/rand", "lower/cipher", "e54 matched", "e54 own", "c3", "c3 copy", "rank1 96", "prior/cipher"]
GAMMA = {"reuse": [1.09, 0.79, 0.76, 0.76, 0.76, 0.85, 0.84, 0.86, 0.77], "new": [0.40, 0.28, 0.14, 0.25, 0.25, 0.24, 0.24, 0.30, 0.38]}
MADE_UP = {
    "A: the gate function carries it (made-up: 4 of 9 apart)": {"reuse": [0.30, 0.12, 0.20, 0.08, 0.08, 0.15, 0.10, 0.22, 0.12],
                                                                "new": [0.10, 0.14, 0.05, 0.10, 0.07, 0.16, 0.16, 0.04, 0.13]},
    "B: the subcomponents carry it (made-up: 9 of 9 apart)": {"reuse": [0.80, 0.60, 0.55, 0.58, 0.58, 0.66, 0.62, 0.64, 0.57],
                                                              "new": [0.30, 0.20, 0.10, 0.18, 0.18, 0.20, 0.20, 0.25, 0.30]},
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = range(len(PAIRS))
    for ax, (title, d) in zip(axes, MADE_UP.items(), strict=True):
        ax.plot(x, GAMMA["reuse"], "o", color="tab:red", alpha=0.35, label="gate function, reuse side (saved)")
        ax.plot(x, GAMMA["new"], "o", color="tab:blue", alpha=0.35, label="gate function, new side (saved)")
        ax.plot(x, d["reuse"], "s", color="tab:red", label="top-k gate, reuse side")
        ax.plot(x, d["new"], "s", color="tab:blue", label="top-k gate, new side")
        ax.set_xticks(list(x), PAIRS, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"rival {title}", fontsize=9)
        ax.axhline(0, color="grey", lw=0.6)
    axes[0].set_ylabel("re-gating share (mean over seeds)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Same subcomponents, two gates: does the separation survive the top-k gate? (neuron version: 6 of 9)", fontsize=10)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
