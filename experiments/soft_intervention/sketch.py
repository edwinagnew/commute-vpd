"""Card sketch for P1.19, made-up numbers only.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/soft_intervention/sketch.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

items = ["planted\ntrigger", "planted\nfloor", "twin\nseed 0", "twin\nseed 1"]
cases = {
    "rival a: the twin reads the plain model's parity": [1.0, 0.0, 0.8, 0.75],
    "rival b: the twin memorised a table, as the fact did": [1.0, 0.0, 0.1, 0.05],
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (title, v) in zip(axes, cases.items(), strict=True):
    ax.scatter(range(4), v, color="#0072B2", marker="o", label="trigger interchange")
    ax.axhline(0.5, color="#D55E00", ls="--", lw=1, label="card gap, 0.5")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(4), items, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.1, 1.1)
axes[0].set_ylabel("share of the predicted move in the push\nthat follows a swap of parity")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("experiments/soft_intervention/figures/sketch.png", dpi=120)
