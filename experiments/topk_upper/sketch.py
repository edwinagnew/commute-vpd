"""The planned figure of P2.42 run 2, drawn with made-up numbers before the run (one panel pair per rival).

    ~/commute-agents/fleet/cpu-run python experiments/topk_upper/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BANDS = ["p90", "p97", "p99"]
# per band: (E217 two-sided, run 2 upper only), each the mean over the 20 trained fine-tunes
MADE_UP = {
    "A: the change is in the strong positive firings": {"overlap": [(0.47, 0.50), (0.29, 0.32), (0.20, 0.22)], "R": [(0.965, 0.968), (0.978, 0.980), (0.986, 0.987)]},
    "B: E217's fall came from GELU's floor tail": {"overlap": [(0.47, 0.75), (0.29, 0.65), (0.20, 0.60)], "R": [(0.965, 0.990), (0.978, 0.993), (0.986, 0.995)]},
}


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, (title, d) in enumerate(MADE_UP.items()):
        for row, (key, label) in enumerate((("overlap", "strong-neuron overlap (Jaccard)"), ("R", "R on the read positions"))):
            ax = axes[row, col]
            ax.plot(BANDS, [a for a, _ in d[key]], "--s", color="grey", label="E217: above p_hi or below p_lo")
            ax.plot(BANDS, [b for _, b in d[key]], "-o", label="run 2: above p_hi only")
            if key == "R":
                ax.axhline(0.9995, color="k", lw=0.8, ls=":", label="full width (E217)")
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("upper percentile defining a strong neuron")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
