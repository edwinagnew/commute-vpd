"""Planned figure for P1.27, drawn with made-up numbers: one panel per rival's outcome. Reading 3 only; per pair, each
larger-side state's survival at the matched size minus its survival at its saved size, with the pair's original
reuse-minus-new gap as a second marker. 0 = size made no difference.

    ~/commute-agents/fleet/cpu-run python experiments/held_model_size_matched/sketch.py
"""

from pathlib import Path

import numpy as np

from commute.figures import FULL_WIDTH, figure, save

PAIRS = ["e62 copy/lookup", "e46 copy/random", "e53 lowercase/cipher", "e54 matched", "e54 own", "c3 rank1/cipher",
         "c3 copy/cipher", "e58 rank1/cipher", "e58 copy/prior"]  # fmt: skip
GAP = [0.004, 0.005, 0.003, 0.002, 0.006, 0.001, 0.007, 0.004, 0.003]  # made-up original gaps
OUTCOMES = {  # made-up mean shift per pair: the matched size moves the new side's S up by about the gap, or not at all
    "rival A: S reads update size (made-up)": [0.8 * g for g in GAP],
    "rival B: S ignores the length (made-up)": [0.0] * len(GAP),
}
rng = np.random.default_rng(0)
fig, axs = figure(FULL_WIDTH, ncols=2)
rows = []
for ax, (title, centres) in zip(axs, OUTCOMES.items(), strict=True):
    for i, (c, g) in enumerate(zip(centres, GAP, strict=True)):
        ys = c + rng.normal(0, 0.0005, 3)
        ax.scatter(ys, [i] * 3, color="#56B4E9", marker="P", s=16, label="shift of one state" if i == 0 else None)
        ax.scatter([g], [i], color="#000000", marker="|", s=40, label="pair's saved-size gap" if i == 0 else None)
        rows += [{"panel": title, "pair": PAIRS[i], "shift": float(y), "gap": g} for y in ys]
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_yticks(range(len(PAIRS)), PAIRS, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("reading 3: S matched minus S saved", fontsize=7)
axs[0].legend(fontsize=6)
save(fig, Path(__file__).parent / "figures" / "sketch.png", n="made-up numbers; 2 to 8 seeds per side", data=rows)
