"""Sketch of P1.40 run 2's figure with made-up numbers: R_A per built item, one panel per rival."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

items = ["x2", "x10", "moved", "rebuilt", "new_few", "new_thin"]
cols = ["tab:blue"] * 4 + ["tab:orange"] * 2
panels = {"rival a: reuse above new": [1.0, 0.99, 0.998, 0.995, 0.90, 0.93],
          "rival b: new edits lose one remainder": [1.0, 0.99, 0.998, 0.995, 0.9995, 0.9996]}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, (t, v) in zip(axs, panels.items(), strict=True):
    ax.bar(items, v, color=cols)
    ax.set_ylim(0.85, 1.005)
    ax.set_title(t, fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
axs[0].set_ylabel("R_A at SAE latents (bar 0.1 KL(F||M))")
fig.suptitle("SKETCH, made-up numbers", fontsize=9)
fig.tight_layout()
fig.savefig(__file__.replace(".py", ".png"), dpi=110)
