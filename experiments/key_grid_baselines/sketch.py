"""P2.52's planned figure drawn with made-up numbers before the run: the top-neuron share (k 64) and the crosscoder's d
on E237's five tasks, one row per rival.

    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid_baselines/sketch.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TASKS = ["cipher", "prior_cipher", "copy", "elicit", "rank1"]  # E237's order, lowest prior first
MADE_UP = {  # per rival, per panel: three seeds per task and the random band (None: undefined)
    "A: sorts by kind, like the share": {
        "top-neuron share, k 64": ([[0.00, 0.002, -0.002], [0.003, 0.0, 0.001], [0.03, 0.033, 0.029], [0.025, 0.028, 0.03], [0.031, 0.027, 0.03]], None),
        "crosscoder d": ([[0.3, 0.27, 0.32], [0.29, 0.31, 0.3], [0.65, 0.6, 0.62], [0.7, 0.68, 0.72], [0.67, 0.7, 0.65]], (0.45, 0.6)),
    },
    "B: no order by kind (seed spread or the prior)": {
        "top-neuron share, k 64": ([[0.015, 0.017, 0.012], [0.011, 0.013, 0.011], [0.026, 0.028, 0.024], [-0.001, -0.004, -0.001], [0.003, -0.003, -0.015]], None),
        "crosscoder d": ([[0.46, 0.38, 0.51], [0.51, 0.42, 0.47], [0.06, 0.3, 0.15], [0.44, 0.5, 0.39], [0.5, 0.42, 0.55]], (0.45, 0.6)),
    },
}


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for row, (rival, panels) in enumerate(MADE_UP.items()):
        for col, (label, (seeds, band)) in enumerate(panels.items()):
            ax = axes[row, col]
            for i, vs in enumerate(seeds):
                if band:
                    ax.fill_between([i - 0.3, i + 0.3], *band, color="grey", alpha=0.25, linewidth=0)
                ax.scatter([i] * 3, vs, color="#D55E00" if col == 0 else "#0072B2", s=20)
            ax.set_xticks(range(len(TASKS)), TASKS, rotation=30, fontsize=7)
            ax.set_ylabel(label)
            ax.set_title(f"rival {rival} (made-up numbers)", fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sketch.png", dpi=110)


if __name__ == "__main__":
    main()
