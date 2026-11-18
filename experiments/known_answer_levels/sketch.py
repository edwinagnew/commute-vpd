"""Planned figure for the known-answer test across levels, drawn with made-up numbers: one panel per rival, R (mean
interchange retention over K, reading A) per edit at each declared level.

    ~/commute-agents/fleet/cpu-run python experiments/known_answer_levels/sketch.py
"""

from pathlib import Path

from commute import figures

LEVELS = ["neuron", "head", "MLP layer", "block"]
MADE_UP = {  # rival -> edit -> R per level
    "Rival A: a coarse level reads the labels": {
        "reuse, small": [0.88, 0.70, 0.68, 0.66],
        "reuse, large": [0.85, 0.62, 0.60, 0.58],
        "new, few units": [0.87, 0.02, 0.01, 0.0],
        "new, thin": [0.995, 0.03, 0.02, 0.0],
    },
    "Rival B: only changed parts are read": {
        "reuse, small": [0.88, 0.0, 0.0, 0.0],
        "reuse, large": [0.84, 0.21, 0.23, 0.25],
        "new, few units": [0.87, 0.0, 0.0, 0.0],
        "new, thin": [0.995, 0.0, 0.0, 0.0],
    },
}
STYLE = {"reuse, small": ("C0", "-"), "reuse, large": ("C0", "--"), "new, few units": ("C3", "-"), "new, thin": ("C3", "--")}

fig, axs = figures.figure(figures.FULL_WIDTH, ncols=2)
rows = []
for ax, (title, edits) in zip(axs, MADE_UP.items(), strict=True):
    for edit, R in edits.items():
        colour, ls = STYLE[edit]
        ax.plot(range(len(LEVELS)), R, color=colour, linestyle=ls, marker="o", label=edit)
        rows += [{"panel": title, "edit": edit, "level": lv, "R": r} for lv, r in zip(LEVELS, R, strict=True)]
    ax.set_xticks(range(len(LEVELS)), LEVELS, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("R, mean share kept over K", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Does some level read the four built edits as built? (sketch, made-up numbers)", fontsize=10)
fig.tight_layout()
figures.save(fig, Path(__file__).parent / "figures" / "sketch.png", n="4 edits, 8 cues each, 512 windows, made-up numbers", data=rows)
