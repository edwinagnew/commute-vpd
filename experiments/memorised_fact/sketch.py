"""Card sketch for P1.16, made-up numbers only.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/memorised_fact/sketch.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

items = ["planted\nodd", "planted\nfact", "fine-tune\nodd (twin)", "fine-tune\nfact"]
cases = {
    "rival a: re-weighting share lower on the fact": {"re-weighting share": [0.95, 0.15, 0.39, 0.10], "rewiring share": [0.98, 0.97, 0.47, 0.50]},
    "rival b: one bit is one direction either way": {"re-weighting share": [0.95, 0.93, 0.39, 0.36], "rewiring share": [0.98, 0.97, 0.47, 0.48]},
}
style = {"re-weighting share": ("#0072B2", "o"), "rewiring share": ("#D55E00", "s")}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (title, series) in zip(axes, cases.items(), strict=True):
    for k, (name, v) in enumerate(series.items()):
        c, m = style[name]
        ax.scatter([i + 0.12 * (2 * k - 1) for i in range(4)], v, color=c, marker=m, label=name)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(4), items, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.1, 1.1)
axes[0].set_ylabel("share of the gap closed")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("experiments/memorised_fact/figures/sketch.png", dpi=120)
