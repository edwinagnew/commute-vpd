"""The planned figure with made-up numbers (card.txt sketch): per side, the best candidate's steering suppression
against its bulk pool's maximum, one panel per rival. Usage: python sketch.py <out.png>."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sides = ["C0b", "C2 A", "C2 B", "rung 1", "rung 5", "E212", "head B"]
rivals = {"rival a: finds the plants": ([0.02, 0.03, 0.02, 0.01, 0.9, 0.6, 0.7], [0.04, 0.05, 0.04, 0.03, 0.2, 0.15, 0.2]),
          "rival b: plants read as noise": ([0.02, 0.06, 0.02, 0.01, 0.1, 0.05, 0.12], [0.04, 0.05, 0.04, 0.03, 0.2, 0.15, 0.2])}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, (title, (cand, pool)) in zip(ax, rivals.items(), strict=True):
    x = range(len(sides))
    a.bar([i - 0.2 for i in x], cand, 0.4, label="best candidate (made up)")
    a.bar([i + 0.2 for i in x], pool, 0.4, color="gray", label="its pool's maximum")
    a.set_xticks(list(x), sides, rotation=30, fontsize=8)
    a.set_title(title, fontsize=9)
ax[0].set_ylabel("suppression of g (nats)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=80)
