"""Figures for cell P1.37 (R at the VPD featurization with site remainders).

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd/plot.py --sketch   (the card's sketch, made-up numbers)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
REUSE = ("x2", "x10", "rebuilt", "moved")
NEW = ("new_few", "new_thin")
COLORS = {"reuse": "#0072B2", "new": "#D55E00"}  # Okabe-Ito, as `figures.METHODS`


def draw_known(ax: Axes, r: dict[str, float], title: str) -> None:
    """R per known-answer item, reuse items above the new edits, with the lowest R1b reuse item marked."""
    items = (*REUSE, *NEW)
    for i, item in enumerate(items):
        side = "reuse" if item in REUSE else "new"
        ax.scatter([r[item]], [i], color=COLORS[side], s=26, linewidths=0, label=side if item in (REUSE[0], NEW[0]) else None)
    ax.set_yticks(range(len(items)), [f"{x} (graft, scores 0)" if x == "moved" else x for x in items])
    ax.invert_yaxis()
    FG.reference(ax, min(r[x] for x in REUSE if x != "moved"), "lowest reuse (x2, x10, rebuilt)")
    ax.set_xlabel("R at the VPD featurization (read positions)")
    ax.set_title(title, fontsize=8, loc="left", pad=12)
    ax.legend(fontsize=7, loc="lower left", frameon=False)


SKETCH = {  # made-up numbers
    "Rival a: both new edits below every reuse item but moved (R1b recovers)":
        {"x2": 0.97, "x10": 0.93, "rebuilt": 0.90, "moved": 0.70, "new_few": 0.62, "new_thin": 0.71},
    "Rival b: new_thin at or above the reuse items (R1b does not recover)":
        {"x2": 0.97, "x10": 0.93, "rebuilt": 0.90, "moved": 0.70, "new_few": 0.80, "new_thin": 0.95},
}


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure((7.5, 3.4), ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, r) in zip(axes, SKETCH.items(), strict=True):
        draw_known(ax, r, title)
        data += [{"panel": title, "item": k, "R": v} for k, v in r.items()]
    fig.suptitle("Planned figure (made-up numbers): does R at VPD put the built new edits below the reuse items?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers; one fine-tune per item, 512 read windows", data=data)


if __name__ == "__main__":
    if "--sketch" in sys.argv[1:]:
        sketch()
