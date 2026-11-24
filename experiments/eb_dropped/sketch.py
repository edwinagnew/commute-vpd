"""The planned figure of P1.57, drawn with made-up numbers before the run (one panel per rival).

    ~/commute-agents/fleet/cpu-run python experiments/eb_dropped/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ARMS = ["E62 copy", "E62 lookup", "grid copy", "grid top", "grid mid", "grid cipher", "grid prior-matched", "randoms"]
MADE_UP = {
    "A: E_b separates them": [0.45, 0.05, 0.40, 0.32, 0.25, 0.08, 0.10, 0.01],
    "B: E_b does not": [0.08, 0.05, 0.06, 0.05, 0.05, 0.04, 0.04, 0.01],
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, v) in zip(axes, MADE_UP.items(), strict=True):
        x = list(range(len(ARMS)))
        for s, d in enumerate((-0.01, 0.0, 0.01)):
            ax.plot([i + 0.15 * (s - 1) for i in x], [y + d for y in v], "o", color=f"C{s}", label=f"seed {s}")
        ax.set_xticks(x, ARMS, rotation=35, ha="right")
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=9)
    axes[0].set_ylabel("E_b over the needed set K")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
