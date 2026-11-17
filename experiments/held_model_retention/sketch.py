"""Planned figure for P1.26 run 1, drawn with made-up numbers: one panel per rival's outcome. Pairs along x, reuse member
minus twin survival up y, one colour per reading, the four seeds as dots, zero as the no-difference line.

    ~/commute-agents/fleet/cpu-run python experiments/held_model_retention/sketch.py
"""

from pathlib import Path

import numpy as np

from commute.figures import FULL_WIDTH, figure, save

PAIRS = ["copy/cipher", "ident/rand", "case", "repeat", "twin/fact"]
READINGS = {"1 neurons and heads": ("#0072B2", "o"), "2 heads and MLP blocks": ("#D55E00", "s"), "3 per-site linear map": ("#009E73", "^")}
rng = np.random.default_rng(0)
OUTCOMES = {  # made-up centres per reading and pair
    "rival A: survival tracks reuse (made-up)": {"1 neurons and heads": [0.20, 0.15, 0.25, 0.18, 0.02],
                                                 "2 heads and MLP blocks": [0.10, 0.08, 0.12, 0.10, 0.0],
                                                 "3 per-site linear map": [0.12, 0.10, 0.15, 0.12, 0.01]},
    "rival B: update size; maps absorb all (made-up)": {"1 neurons and heads": [0.01, 0.01, 0.20, 0.08, 0.0],
                                                        "2 heads and MLP blocks": [0.0, 0.01, 0.10, 0.04, 0.0],
                                                        "3 per-site linear map": [0.0, 0.0, 0.002, 0.001, 0.0]},
}  # fmt: skip

fig, axs = figure(FULL_WIDTH, ncols=2)
rows = []
for ax, (title, centres) in zip(axs, OUTCOMES.items(), strict=True):
    for k, (reading, (color, marker)) in enumerate(READINGS.items()):
        for i, c in enumerate(centres[reading]):
            ys = c + rng.normal(0, 0.02, 4)
            ax.scatter([i + 0.2 * (k - 1)] * 4, ys, color=color, marker=marker, s=16, label=reading if i == 0 else None)
            rows += [{"panel": title, "reading": reading, "pair": PAIRS[i], "seed": s, "gap": float(y)} for s, y in enumerate(ys)]
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(range(len(PAIRS)), PAIRS, fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("survival S, reuse member minus twin", fontsize=7)
axs[0].legend(fontsize=6)
save(fig, Path(__file__).parent / "figures" / "sketch.png", n="made-up numbers; 4 seeds per pair; 512 read windows per item", data=rows)
