"""Figures for cell P1.37 (R at the VPD featurization with site remainders).

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd/plot.py --sketch   (the card's sketch, made-up numbers)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/kstar_vpd/plot.py            (after score.py)
"""

from __future__ import annotations

import json
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
    ax.set_yticks(range(len(items)), [f"{x}*" if x == "moved" else x for x in items])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    FG.reference(ax, min(r[x] for x in REUSE if x != "moved"), "lowest of x2, x10, rebuilt")
    ax.set_xlabel("R at the VPD featurization (read positions)")
    ax.set_title(title, fontsize=8, loc="left", pad=14)
    ax.legend(fontsize=7, loc="upper left", frameon=False)


SKETCH = {  # made-up numbers
    "Rival a: R1b recovers":
        {"x2": 0.97, "x10": 0.93, "rebuilt": 0.90, "moved": 0.70, "new_few": 0.62, "new_thin": 0.71},
    "Rival b: new_thin at the reuse level":
        {"x2": 0.97, "x10": 0.93, "rebuilt": 0.90, "moved": 0.70, "new_few": 0.80, "new_thin": 0.95},
}


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure((7.5, 3.4), ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, r) in zip(axes, SKETCH.items(), strict=True):
        draw_known(ax, r, title)
        data += [{"panel": title, "item": k, "R": v} for k, v in r.items()]
    fig.suptitle("Planned figure (made-up numbers): are the built new edits below the reuse items?", fontsize=9)
    FG.save(fig, out / "sketch.png", n="made-up numbers; one fine-tune per item. *moved: its 8 appended neurons are graft variables scoring 0, "
            "so it is predicted low", data=data)  # fmt: skip


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    """From results/summary.json (score.py): R per known item, and each E187 pair's R gap against minus its size gap."""
    s = json.loads((results / "summary.json").read_text())
    r = {i: v["R_read"] for i, v in s["per_item"].items() if i in (*REUSE, *NEW)}
    fig, (a, b) = FG.figure(FG.FULL_WIDTH, ncols=2)
    if len(r) == len(REUSE) + len(NEW):
        draw_known(a, r, f"Known items: R1b {s['R1b']['call']} (band {s['band']['band']:.3g})")
    p = s["pairs"]
    pts = [(p["minus_size_gap"][k], v) for k, v in p["gaps"]["read"].items() if v is not None and p["minus_size_gap"].get(k) is not None]
    b.scatter([x for x, _ in pts], [y for _, y in pts], color=COLORS["reuse"], s=24, linewidths=0)
    b.set_xlabel("minus update-size gap (reuse - new)")
    b.set_ylabel("R gap, reuse - new (read positions)")
    b.set_title(f"E187 pairs: {p['vs_minus_size']['call']} with size", fontsize=8, loc="left")
    fig.suptitle("R at the VPD featurization: known items and E187's pairs", fontsize=9)
    data = [{"item": i, "R": v} for i, v in r.items()] + [{"pair_minus_size_gap": x, "pair_R_gap": y} for x, y in pts]
    FG.save(fig, out / "r_known_and_pairs.png", n="one fine-tune per known item (*moved: graft variables score 0); E187's nine pairs, every saved state",
            data=data)  # fmt: skip


if __name__ == "__main__":
    if "--sketch" in sys.argv[1:]:
        sketch()
    else:
        main()
