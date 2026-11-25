"""Card sketch for P1.21, made-up numbers only.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/search_recovery/sketch.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

items = ["planted\nconstant", "planted\ntrigger", "cue", "twin\ns0", "twin\ns1", "fact\ns0", "fact\ns1"]
search = [0.72, 0.54, 0.44, 0.64, 0.66, 0.74, 0.74]  # E184's best at the filed rung (real values, for orientation)
cases = {
    "rival a: the search misses an edit the class holds": [0.97, 0.93, 0.91, 0.92, 0.92, 0.80, 0.80],
    "rival b: the edit is not in the class at its filed rung": [0.55, 0.50, 0.30, 0.62, 0.63, 0.70, 0.71],
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
x = range(len(items))
for ax, (title, v) in zip(axes, cases.items(), strict=True):
    ax.scatter([i - 0.1 for i in x], search, color="#999999", label="E184 search")
    ax.scatter([i + 0.1 for i in x], v, color="#0072B2", label="oracle (built edit by hand)")
    ax.axhline(0.9, color="#D55E00", lw=0.8, ls="--")
    ax.set_xticks(list(x), items, fontsize=7)
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel("closure at the filed rung (<= 16 names)")
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("experiments/search_recovery/figures/sketch.png", dpi=120)
