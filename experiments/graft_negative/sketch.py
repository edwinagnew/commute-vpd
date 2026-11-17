"""Planned figure for P1.26 run 2, drawn with made-up numbers: one panel per rival's outcome. The three readings along x,
survival S of the built F (M plus the grafted head) up y, with S = 0, the arithmetic value of readings 1 and 2 when the
graft is left unpaired, as a line; reading 3's graft-relative residual at the graft's site as a second marker.

    ~/commute-agents/fleet/cpu-run python experiments/graft_negative/sketch.py
"""

from pathlib import Path

from commute.figures import FULL_WIDTH, figure, save

READINGS = ["1 neurons and heads", "2 heads and MLP blocks", "3 per-site linear map"]
OUTCOMES = {  # made-up S per reading, and reading 3's residual of the graft's own write at its site
    "rival A: the graft's route is new to M (made-up)": ([0.0, 0.0, 0.06], 0.91),
    "rival B: M's heads already carry the token (made-up)": ([0.0, 0.0, 0.93], 0.05),
}

fig, axs = figure(FULL_WIDTH, ncols=2)
rows = []
for ax, (title, (S, res)) in zip(axs, OUTCOMES.items(), strict=True):
    ax.bar(range(3), S, color=["#0072B2", "#D55E00", "#009E73"], width=0.6)
    ax.scatter([2.35], [res], color="#000000", marker="x", s=24, label="reading 3: share of the graft's write left unfitted")
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3), READINGS, fontsize=6)
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("survival S of the grafted model", fontsize=7)
    rows += [{"panel": title, "reading": r, "S": s} for r, s in zip(READINGS, S, strict=True)]
    rows.append({"panel": title, "reading": "3 graft-relative residual", "S": res})
axs[0].legend(fontsize=6, loc="upper left")
save(fig, Path(__file__).parent / "figures" / "sketch.png", n="made-up numbers; one built graft; 512 read windows, last position", data=rows)
