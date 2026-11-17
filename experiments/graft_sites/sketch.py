"""Planned figure for P1.28, drawn with made-up numbers: one panel per rival's outcome. One row per graft, survival S of
the grafted model across: the per-site map (reading 3, the main reading) and one gain per unit (reading 1), with the
lowest old-task fine-tune of E197 and E201's graft under reading 3 as reference lines.

    ~/commute-agents/fleet/cpu-run python experiments/graft_sites/sketch.py
"""

from pathlib import Path

from commute.figures import FULL_WIDTH, figure, reference, save

GRAFTS = ["head, last layer, k 16 (rebuild)", "MLP neurons, last layer", "head, layer 1, k 16", "head, last layer, k 12",
          "head, last layer, k 8", "head, last layer, k 4"]  # fmt: skip
OUTCOMES = {  # made-up reading-3 S per graft
    "rival A: no site's map rebuilds new machinery (made-up)": [0.12, 0.15, 0.10, 0.14, 0.17, 0.22],
    "rival B: a map absorbs what M's units carry (made-up)": [0.13, 0.95, 0.11, 0.16, 0.20, 0.30],
}

fig, axs = figure(FULL_WIDTH, ncols=2)
rows = []
for ax, (title, S) in zip(axs, OUTCOMES.items(), strict=True):
    y = list(range(len(GRAFTS)))
    ax.scatter(S, y, color="#009E73", s=24, label="per-site map (reading 3)")
    ax.scatter([0.0] * len(S), y, color="#0072B2", marker="s", s=18, label="one gain per unit (reading 1)")
    reference(ax, 0.984, "lowest old-task fine-tune")
    reference(ax, 0.134, "E201's graft")
    ax.set_yticks(y, GRAFTS if ax is axs[0] else [""] * len(y), fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("survival S of the grafted model", fontsize=7)
    ax.set_title(title, fontsize=7)
    rows += [{"panel": title, "graft": g, "S3": s, "S1": 0.0} for g, s in zip(GRAFTS, S, strict=True)]
axs[0].legend(fontsize=6, loc="lower right")
save(fig, Path(__file__).parent / "figures" / "sketch.png", n="made-up numbers; one build per graft; 512 read windows, last position", data=rows)
