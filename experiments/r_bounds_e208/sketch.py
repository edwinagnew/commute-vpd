"""Planned figure for T1 (bounds on R from E208's saved per-part scores), drawn with made-up numbers: one panel per
rival's outcome, the worst R a circuit of k parts can have (mean of the k lowest scores) against k, one line per item.

    ~/commute-agents/fleet/cpu-run python experiments/r_bounds_e208/sketch.py
"""

from pathlib import Path

import numpy as np

from commute import figures

K = 2 ** np.arange(13)
MADE_UP = {  # panel -> item -> 1 - worst R at k = 1, ..., 4096 (a power law down to the all-part mean)
    "Rival A: saturation is the level's (cos^2)": {"reuse member": (0.06, 2e-4), "its twin": (0.07, 3e-4)},
    "Rival B: saturation is cos^2's (size-aware)": {"reuse member": (0.12, 0.018), "its twin": (0.16, 0.030)},
}

fig, axs = figures.figure(figures.FULL_WIDTH, ncols=2)
rows = []
for ax, (title, items) in zip(axs, MADE_UP.items(), strict=True):
    for (name, (lo_k1, lo_all)), style in zip(items.items(), ("-", "--"), strict=True):
        y = lo_k1 * (lo_all / lo_k1) ** (np.log2(K) / 12)
        ax.plot(K, y, style, color="#1f77b4", label=name)
        rows += [{"panel": title, "item": name, "k": int(k), "one_minus_R": float(v)} for k, v in zip(K, y, strict=True)]
    ax.axhline(0.01, color="#777777", linestyle=":", linewidth=0.8, label="R = 0.99")
    ax.set(xscale="log", yscale="log", xlabel="parts in the circuit, k (lowest scores first)", ylabel="1 - worst R")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle("How low can R go for a circuit of k parts? (sketch, made-up numbers)", fontsize=10)
figures.save(fig, Path(__file__).parent / "figures" / "sketch.png", n="made-up numbers; the run draws 20 fine-tunes, four distances", data=rows)
