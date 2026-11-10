"""E149's figure, from results/stage1.json's score only (no model).

votes.png: per built pair (rows), each reading's call against the share's direction: +1 the reading puts the sides the
way the share does (the side built to reuse higher, on every built pair), 0 no call (prior gap under the bar; steering not
separated), -1 the opposite way. The last two columns combine the prior with steering in its two forms: +1 tracked,
0 conflict (the two point opposite ways), -1 missed (neither points the share's way). x: not read.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/prior_steering/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
COLS = {"share": "share", "prior": "prior", "steer_filed": "steer,\nsearch", "steer_s1": "steer,\ns = 1",
        "together_filed": "both,\nsearch", "together_s1": "both,\ns = 1"}  # fmt: skip
TOGETHER = {"tracked": 1, "conflict": 0, "missed": -1}
LABEL = "+1 the share's way, 0 no call or conflict,\n-1 the other way or missed"


def cell(r: Mapping[str, Any], col: str) -> float:
    v = {"share": r["share"], "prior": r["prior"]["vote"], "steer_filed": r["steering"]["filed"]["vote"],
         "steer_s1": r["steering"]["strength1"]["vote"], "together_filed": TOGETHER.get(r["together"]["filed"]),
         "together_s1": TOGETHER.get(r["together"]["strength1"])}[col]  # fmt: skip
    return math.nan if v is None else float(v)


def panel(ax: Axes, score: Mapping[str, Any], title: str) -> list[dict[str, Any]]:
    pairs = score["pairs"]
    FG.gap_heatmap(ax, [[cell(r, c) for c in COLS] for r in pairs.values()], [r["name"] for r in pairs.values()], list(COLS.values()),
                   label=LABEL, vmax=1.0, fmt=".0f")  # fmt: skip
    ax.set_title(title)
    return [{"pair": p, "column": c, "vote": cell(r, c), "prior_gap_decades": r["prior"]["gap"], "steer_filed_gap": r["steering"]["filed"].get("gap"),
             "steer_s1_gap": r["steering"]["strength1"].get("gap")} for p, r in pairs.items() for c in COLS]  # fmt: skip


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    score = json.loads((results / "stage1.json").read_text())["score"]
    o = score["outcome"]
    fig, ax = FG.figure(FG.FULL_WIDTH)
    data = panel(ax, score, f"Prior with steering vs the share: {o['strength1']} at s = 1, {o['filed']} searched")
    n = "2 to 8 seeds a side for share and steering; the prior is the original model's, one value per map; bar 0.5 decades"
    FG.save(fig, figures / "votes.png", n=n, data=data)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
