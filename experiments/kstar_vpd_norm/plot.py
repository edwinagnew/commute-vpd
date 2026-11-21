"""Figures for cell P1.37 run 2 (size-normalised R at the VPD featurization).

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd_norm/plot.py --sketch   (the card's sketch, made-up numbers)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd_norm/plot.py            (after run.py's summary)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
REUSE = ("x2", "x10", "rebuilt", "moved")
NEW = ("new_few", "new_thin")
COLORS = {"reuse": "#0072B2", "new": "#D55E00"}  # Okabe-Ito, as `figures.METHODS`


def draw_known(ax: Axes, n: Mapping[str, float | None], band: float, title: str) -> None:
    """N per known-answer item (1 - mean turn share over changed K* members), with the highest new edit plus the band."""
    items = (*REUSE, *NEW)
    for i, item in enumerate(items):
        side = "reuse" if item in REUSE else "new"
        v = n[item]
        if v is not None:
            ax.scatter([v], [i], color=COLORS[side], s=28, linewidths=0, label=side if item in (REUSE[0], NEW[0]) else None)
    ax.set_yticks(range(len(items)), list(items))
    ax.invert_yaxis()
    top = max(v for k, v in n.items() if k in NEW and v is not None)
    ax.axvline(top, color=COLORS["new"], linewidth=0.8, linestyle="--", label="highest new edit")
    ax.axvspan(top, top + band, color=COLORS["new"], alpha=0.15, linewidth=0, label="plus the band: x10 must sit right of it")
    ax.set_xlabel("N = 1 - mean turn share t over the changed parts of K*")
    ax.set_title(title, fontsize=8, loc="left", pad=14)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)


SKETCH = {  # made-up numbers
    "Rival a: x10 above both new edits": {"x2": 0.85, "x10": 0.80, "rebuilt": 0.55, "moved": 0.82, "new_few": 0.40, "new_thin": 0.50},
    "Rival b: x10 at or below a new edit": {"x2": 0.50, "x10": 0.35, "rebuilt": 0.45, "moved": 0.52, "new_few": 0.40, "new_thin": 0.55},
}


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure((7.5, 3.4), ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, n) in zip(axes, SKETCH.items(), strict=True):
        draw_known(ax, n, 0.08, title)
        data += [{"panel": title, "item": k, "N": v} for k, v in n.items()]
    fig.suptitle("Planned figure (made-up numbers): does x10 read above both built new edits once size is divided out?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers; one fine-tune per item; band from the two halves of D", data=data)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    s = json.loads((results / "summary.json").read_text())
    fig, ax = FG.figure((7.0, 3.4))
    draw_known(ax, s["known"]["N"], s["band"]["N"], "N per known-answer item (weight norms)")
    FG.save(fig, out / "n_known.png", n="one fine-tune per item; grafts (moved) are left out of t", data=[{"item": k, "N": v} for k, v in s["known"]["N"].items()])


if __name__ == "__main__":
    if "--sketch" in sys.argv[1:]:
        sketch()
    else:
        main()
