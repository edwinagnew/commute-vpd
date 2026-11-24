"""Planned figure for the E62 copy-windows read (cell P1.57), made-up numbers, one panel per rival.

    ~/commute-agents/fleet/cpu-run python experiments/e62_copy_windows/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
MADE_UP = {"A: the mix diluted it": {"all of D": (0.038, 0.042), "copy windows": (0.30, 0.05)},
           "B: no separation either way": {"all of D": (0.038, 0.042), "copy windows": (0.045, 0.044)}}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, d) in zip(axes, MADE_UP.items(), strict=True):
        for k, (where, (c, lk)) in enumerate(d.items()):
            ax.bar([k - 0.18, k + 0.18], [c, lk], width=0.34, color=["C0", "C1"], label=["copy", "lookup"] if k == 0 else None)
        ax.set_xticks([0, 1], list(d))
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=9)
    axes[0].set_ylabel("E_b over K (mean of 3 seeds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
