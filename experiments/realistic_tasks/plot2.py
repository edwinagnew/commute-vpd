"""Run 2 (E211) figure: each method's gap, reuse side minus new side, per pair and size, from score2.py's output.

    ~/commute-agents/fleet/cpu-run python experiments/realistic_tasks/plot2.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from matplotlib.patches import Rectangle

from commute import figures

HERE = Path(__file__).parent
ROWS = [  # (score2 row key, label): the scored unit of each pair
    ("L1/L3 size0 contrast", "topic vs coin, 0.005"),
    ("L1/L3 size1 contrast", "topic vs coin, 0.010"),
    ("T2/T3 size0 lo", "French vs Yoruba, 0.00875"),
    ("T2/T3 size1 lo", "French vs Yoruba, 0.0175"),
    ("T1/T4 size0 lo", "lowercase vs reversed, 0.00875"),
    ("T1/T4 size1 lo", "lowercase vs reversed, 0.0175"),
]
COLS = [
    ("share", "share"),
    ("untrained", "untrained\nshare"),
    ("steering", "steering"),
    ("neuron", "neurons\nk 64"),
    ("k_half", "k_half"),
    ("erank", "erank"),
    ("prior", "prior"),
]


def main() -> None:
    score = json.loads((HERE / "results" / "run2_score.json").read_text())["rows"]
    values, records, apart = [], [], []
    for i, (key, row_label) in enumerate(ROWS):
        row = []
        for j, (method, col_label) in enumerate(COLS):
            cell = score[key]["cells"][method]
            gap = cell["gap"]
            gap = math.nan if gap is None else float(gap)
            row.append(gap)
            if cell["call"] == "apart":
                apart.append((i, j))
            records.append({"pair_size": row_label, "method": col_label.replace("\n", " "), "gap": gap,
                            "call": cell["call"], "filed": cell["filed"]})
        values.append(row)
    fig, ax = figures.figure(figures.FULL_WIDTH)
    figures.gap_heatmap(ax, values, [r for _, r in ROWS], [c for _, c in COLS], vmax=0.5,
                        label="gap, + = reuse side higher (erank: log10 ratio;\nk_half: log10 new/reuse, + = fewer; prior: log-odds)")
    for i, j in apart:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6))
    ax.set_xlabel("method (boxed: seed ranges apart; × = unreadable, a side's gain under 1)")
    ax.set_title("The share reads apart only on lowercase vs reversed; French vs Yoruba is unreadable")
    figures.save(fig, HERE / "figures" / "run2_gaps.png", n="3 seeds per side per size; prior one value per side",
                 data=records)


if __name__ == "__main__":
    main()
