"""Card sketch for P1.15, made-up numbers only.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/difficulty_ladder/sketch.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rungs = ["last number odd", "sum of two", "sum of three"]
cases = {
    "rival a: share falls where the MLP probe falls": {"share": [0.75, 0.45, 0.15], "linear probe": [1.0, 0.1, 0.0], "MLP probe": [1.0, 0.6, 0.2]},
    "rival b: share mirrors the linear probe only": {"share": [0.70, 0.60, 0.58], "linear probe": [1.0, 0.1, 0.0], "MLP probe": [1.0, 1.0, 1.0]},
}
style = {"share": ("#0072B2", "o", "-"), "linear probe": ("#777777", "s", "--"), "MLP probe": ("#D55E00", "^", ":")}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (title, series) in zip(axes, cases.items(), strict=True):
    for name, v in series.items():
        c, m, ls = style[name]
        ax.plot(range(3), v, color=c, marker=m, linestyle=ls, label=name)
    ax.set_xticks(range(3), rungs)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.1, 1.1)
axes[0].set_ylabel("rescaled share (floor 0, ceiling 1)\nprobe: (accuracy - 0.5) / 0.5")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("experiments/difficulty_ladder/figures/sketch.png", dpi=120)
