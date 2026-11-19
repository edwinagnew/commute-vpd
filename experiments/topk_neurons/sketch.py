"""The planned figure of P2.42 run 1, drawn with made-up numbers before the run (one panel per rival).

    ~/commute-agents/fleet/cpu-run python experiments/topk_neurons/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BANDS = ["width", "p90/10", "p97/3", "p99/1"]
PAIRS = ["copy/cipher", "ident/rand", "case", "repeat"]
MADE_UP = {"A: saturates": {"reuse": [0.9999, 0.998, 0.997, 0.997], "twin": [0.9998, 0.997, 0.996, 0.996]},
           "B: strong firings carry the change": {"reuse": [0.9999, 0.985, 0.96, 0.93], "twin": [0.9998, 0.97, 0.92, 0.88]}}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, d) in zip(axes, MADE_UP.items(), strict=True):
        for side, style in (("reuse", "-o"), ("twin", "--s")):
            ax.plot(BANDS, d[side], style, label=f"{side} member (mean over {len(PAIRS)} counted pairs)")
        ax.axhline(0.99, color="grey", lw=0.8, ls=":")
        ax.set_title(f"rival {title} (made-up numbers)")
        ax.set_xlabel("featurization (which activations count as features)")
    axes[0].set_ylabel("R on the read positions, K* by mean ablation")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
