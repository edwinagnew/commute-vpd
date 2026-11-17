"""Planned figure for P1.25 run 2, drawn with made-up numbers: one panel per rival (level A, mean over orders).

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention_pairs/sketch.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

pairs = ["copy/cipher", "ident/rand", "case", "repeat"]
outcomes = {"rival A: reuse keeps rules (made-up)": ([0.999, 0.998, 0.997, 0.997], [0.97, 0.96, 0.95, 0.96]),
            "rival B: reading tracks update size (made-up)": ([0.9990, 0.9990, 0.9985, 0.9985], [0.9989, 0.9988, 0.990, 0.997])}  # fmt: skip
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (title, (reuse, twin)) in zip(axs, outcomes.items(), strict=True):
    ys = range(len(pairs))
    ax.scatter(reuse, [y - 0.1 for y in ys], color="#0072B2", label="reuse member")
    ax.scatter(twin, [y + 0.1 for y in ys], color="#D55E00", label="twin")
    ax.set_yticks(list(ys), pairs, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("level A retention for the contrast", fontsize=8)
    ax.legend(fontsize=7)
axs[0].invert_yaxis()
fig.tight_layout()
(Path(__file__).parent / "figures").mkdir(exist_ok=True)
fig.savefig(Path(__file__).parent / "figures" / "sketch.png", dpi=120)
