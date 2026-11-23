"""The planned figure of the head-(2,4) built edit (cell P1.57), made-up numbers, one panel per rival.

    ~/commute-agents/fleet/cpu-run python experiments/head_edit/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EDITS = ["V x2", "O x2", "QK x2"]
MADE_UP = {"A: E_b sees one-head reuse": {"VPD": [0.70, 0.75, 0.10], "neurons/heads": [0.99, 0.99, 0.08]},
           "B: VPD spreads it": {"VPD": [0.12, 0.60, 0.05], "neurons/heads": [0.99, 0.99, 0.08]}}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, d) in zip(axes, MADE_UP.items(), strict=True):
        for k, (feat, v) in enumerate(d.items()):
            ax.bar([i + 0.38 * k for i in range(3)], v, width=0.36, label=feat)
        ax.axhline(0.761, ls=":", color="k", label="built x2 at VPD (E268)")
        ax.set_xticks([i + 0.19 for i in range(3)], EDITS)
        ax.set_title(f"rival {title}\n(made-up numbers)", fontsize=9)
    axes[0].set_ylabel("E_b over the needed set K")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
