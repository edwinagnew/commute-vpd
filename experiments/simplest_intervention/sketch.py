"""Card sketch for P1.20, made-up numbers only.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/simplest_intervention/sketch.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

items = ["L1 one\nneuron", "L2 neuron\ntriggered", "L4 random\nbit", "twin\ns0", "twin\ns1", "fact\ns0", "fact\ns1"]
cases = {
    "rival a: the twin reuses M's parity, the fact needs a new direction": [1, 2, 4, 2, 2, 3, 3],
    "rival b: both fine-tunes need a new direction (same rung)": [1, 2, 4, 3, 3, 3, 3],
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, (title, v) in zip(axes, cases.items(), strict=True):
    ax.scatter(range(len(items)), v, color="#0072B2")
    ax.set_xticks(range(len(items)), items, fontsize=7)
    ax.set_yticks(range(5), ["0 none", "1 re-set", "2 new trigger", "3 new direction", "4 from scratch"], fontsize=7)
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel("lowest rung closing 0.9 of KL(F||M)")
fig.tight_layout()
fig.savefig("experiments/simplest_intervention/figures/sketch.png", dpi=120)
