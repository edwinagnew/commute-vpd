"""Planned figure for the rho-on-saved-items cell, drawn with made-up numbers: one panel per rival, the main reading's
gap (reuse member's rho minus its twin's, mean over 5 orders) per counted pair, both seeds as points.

    ~/commute-agents/fleet/cpu-run python experiments/rho_saved_items/sketch.py
"""

from pathlib import Path

from commute import figures

PAIRS = ["copy/cipher (x1.05-1.09)", "ident/random map (x1.13)", "case (x2.5-2.7)", "repeat (x1.4-1.6)"]
MADE_UP = {  # rival -> pair -> (seed 0, seed 1) gaps
    "Rival A: heads carry the new table": [(1.5e-4, 1.2e-4), (1.4e-3, 1.2e-3), (2.6e-3, 2.2e-3), (6e-4, 7e-4)],
    "Rival B: gaps follow update size": [(4e-5, 2e-5), (6e-5, 7e-5), (2.4e-3, 2.0e-3), (5e-4, 6e-4)],
}

fig, axs = figures.figure(figures.FULL_WIDTH, ncols=2)
rows = []
for ax, (title, gaps) in zip(axs, MADE_UP.items(), strict=True):
    figures.dots(ax, dict(zip(PAIRS, gaps, strict=True)), method="neuron", xlabel="reuse rho minus twin rho")
    figures.reference(ax, 1e-4, "E196 order sd")
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    rows += [{"panel": title, "pair": p, "seed": s, "gap": g} for p, gs in zip(PAIRS, gaps, strict=True) for s, g in enumerate(gs)]
fig.suptitle("Does the note's rho rank the pairs by update size? (sketch, made-up numbers)", fontsize=10)
fig.tight_layout()
figures.save(fig, Path(__file__).parent / "figures" / "sketch.png", n="2 seeds per pair, 4 counted pairs, made-up numbers", data=rows)
