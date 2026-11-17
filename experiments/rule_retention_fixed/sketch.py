"""Planned figure for P1.25 run 1, drawn with made-up numbers: one panel per rival."""

from pathlib import Path

import matplotlib.pyplot as plt

items = ["do-nothing", "x2", "x10", "moved copy", "rebuilt"]
outcomes = {"rival A: rebuilt rewires": [1, 1, 1, 1, 0.62], "rival B: rebuilt is gain only": [1, 1, 1, 1, 0.995]}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (title, ys) in zip(axs, outcomes.items(), strict=True):
    ax.bar(items, ys, color=["#999"] * 4 + ["#c44"])
    ax.set_title(title + " (made-up numbers)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
axs[0].set_ylabel("retention for the contrast")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figures" / "sketch.png", dpi=120)
