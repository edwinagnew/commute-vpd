import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
out = sys.argv[1]
sites = ["L0 attn", "L0 mlp", "L1 attn", "L1 mlp", "L2 attn", "L2 mlp", "L3 attn", "L3 mlp"]
x = np.arange(len(sites))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, (title, ft) in zip(axes, [("rival a (made up): fine-tunes stay on the dictionary", 0.005),
                                   ("rival b (made up): fine-tunes fall off the dictionary", 0.06)]):
    base = 0.85 + 0.03 * np.sin(x)
    ax.plot(x, base, "o-", label="base (held-out Pile)")
    ax.plot(x, base - ft - 0.01 * np.cos(x) ** 2, "s--", label="fine-tunes (median of 74)")
    ax.set_xticks(x, sites, rotation=45, fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("FVE")
axes[0].legend(fontsize=7)
fig.suptitle("SKETCH, made-up numbers: FVE per site, base vs fine-tunes", fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=100)
