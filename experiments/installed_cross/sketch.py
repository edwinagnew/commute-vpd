"""P2.65's planned figure drawn with made-up numbers (the card's sketch): per reading, the AUC of the 9 diagonal (reuse)
against the 18 off-diagonal (new) states at matched size, beside the matched Gaussians' AUC, under each rival."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures" / "sketch.png"
names = ["steering", "neuron k64", "untrained SVD", "erank", "k_half", "Pile KL", "ADL D"]
rivals = {"a: readings follow training history": [0.95, 0.9, 0.8, 0.7, 0.75, 0.6, 0.55],
          "b: readings follow update size only": [0.55, 0.5, 0.45, 0.5, 0.55, 0.5, 0.45]}  # fmt: skip
gauss = [0.5, 0.52, 0.48, 0.5, 0.47, 0.5, 0.53]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (title, real) in zip(axes, rivals.items(), strict=True):
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], real, 0.4, label="real updates (9 vs 18)", color="#3b6ea5")
    ax.bar([i + 0.2 for i in x], gauss, 0.4, label="matched Gaussians", color="#bbbbbb")
    ax.axhline(0.85, ls="--", c="k", lw=1)
    ax.axhline(0.5, ls=":", c="k", lw=1)
    ax.set_xticks(list(x), names, rotation=35, ha="right")
    ax.set_title(f"rival {title}\n(MADE-UP NUMBERS)", fontsize=9)
axes[0].set_ylabel("AUC reuse vs new, matched size")
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110)
print(OUT)
