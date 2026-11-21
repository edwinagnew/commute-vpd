"""Figure for cell P1.42: each measure's six built edits on one axis, reuse-like to the right.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/r_reanalysis/plot.py --sketch   (made-up numbers)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/r_reanalysis/plot.py            (after run.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
REUSE = ("x2", "x10", "moved", "rebuilt")
NEW = ("new_few", "new_thin")
COLORS = {"reuse": "#0072B2", "new": "#D55E00"}
MARK = {"x2": "o", "x10": "s", "moved": "^", "rebuilt": "D", "new_few": "o", "new_thin": "s"}
ROWS = {  # summary key -> (label, higher value = reuse-like)
    "R": ("R as filed (mean grade over the circuit)", True),
    "Rw_K": ("weighted by behaviour, over the circuit", True),
    "Rw_all": ("weighted by behaviour, every variable", True),
    "R_size_removed": ("R with update size removed", True),
    "Rw_K_size_removed": ("weighted R with update size removed", True),
    "update_size": ("plain update size (smaller = reuse)", False),
    "kl_fm": ("behaviour change from M (smaller = reuse)", False),
}


def scaled(v: dict[str, float], up: bool) -> dict[str, float]:
    """Oriented so right is reuse-like, 0 at the most reuse-like new edit, 1 = the six items' full range."""
    o = {i: (x if up else -x) for i, x in v.items()}
    span = max(o.values()) - min(o.values())
    top_new = max(o[i] for i in NEW)
    return {i: (x - top_new) / span for i, x in o.items()}


def draw(known: dict[str, dict[str, float]], title: str, out: Path, n: str) -> None:
    fig, ax = FG.figure((7.5, 3.8))
    data: list[dict[str, Any]] = []
    rows = [k for k in ROWS if k in known]
    for y, key in enumerate(rows):
        sc = scaled(known[key], ROWS[key][1])
        for i, x in sc.items():
            side = "reuse" if i in REUSE else "new"
            ax.scatter([x], [y], color=COLORS[side], marker=MARK[i], s=30, linewidths=0, zorder=3,
                       label=f"{i} ({'built as reuse' if side == 'reuse' else 'built as new'})" if y == 0 else None)
            data.append({"measure": key, "item": i, "value": known[key][i], "scaled": x})
    ax.set_yticks(range(len(rows)), [ROWS[k][0] for k in rows], fontsize=7)
    ax.invert_yaxis()
    FG.reference(ax, 0.0, "most reuse-like new edit")
    ax.set_xlabel("position against the most reuse-like new edit, in units of the six edits' range (right = reuse-like)", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    FG.save(fig, out, n=n, data=data)


SKETCH = {"R": {"x2": 0.999, "x10": 0.975, "moved": 0.999, "rebuilt": 0.995, "new_few": 0.994, "new_thin": 0.996},
          "Rw_K": {"x2": 0.99, "x10": 0.98, "moved": 0.99, "rebuilt": 0.985, "new_few": 0.95, "new_thin": 0.96},
          "update_size": {"x2": 0.016, "x10": 0.15, "moved": 0.016, "rebuilt": 0.048, "new_few": 0.029, "new_thin": 0.014}}


def main() -> int:
    if "--sketch" in sys.argv:
        draw(SKETCH, "Planned figure (made-up numbers): a measure passes when every blue mark is right of the line\n"
             "rival a: the weighted row passes; rival b: it looks like the filed row", HERE / "figures/sketch.png", "made-up numbers; one fine-tune per item")  # fmt: skip
        return 0
    s = json.loads((HERE / "results/summary.json").read_text())
    known = {k: v["values"] for k, v in s["known"].items() if k in ROWS and v.get("read", True)}
    passing = [ROWS[k][0] for k, v in s["known"].items() if k in ROWS and v.get("orders")]
    draw(known, "Which measures put all four built reuse edits right of both built new edits?\n"
         f"passing: {'; '.join(passing) or 'none'}", HERE / "figures/known_orders.png",
         "one fine-tune per item (six built edits, E226's saved readings); blue = built as reuse, orange = built as new")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
