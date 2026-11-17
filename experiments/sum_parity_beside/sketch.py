"""Card sketch for P1.14, made-up numbers only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = ["planted ceiling", "control: last number odd (saved)", "new: sum parity", "planted floor"]
cases = {"rival a: new reads clearly below": [0.97, 0.47, 0.22, 0.20], "rival b: no separation": [0.97, 0.47, 0.44, 0.43]}
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, (title, v) in zip(axes, cases.items()):
    for i, x in enumerate(v):
        ax.plot([x - 0.01, x + 0.01], [i, i], "o", color="C0" if i in (1, 2) else "0.5")
    ax.axvline(0.5, ls="--", color="k", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("best found rewiring share, one site (dots: seeds)")
axes[0].set_yticks(range(4), rows)
fig.tight_layout()
fig.savefig("experiments/sum_parity_beside/figures/sketch.png", dpi=120)
