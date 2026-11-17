"""E137's one figure, from results/stage1.json and results/stage2.json only (no model).

  found_vs_witness.png  per pair, the found G (k = 4, best site, best of three starts) and the found H (best site) beside the
                        witness's G and H, against the 0.9 search bar and 0 (the identity map). The question: does the blind
                        search recover the known answers, and in what order does it put the pairs?

Usage (repo root):  ~/commute-agents/fleet/cpu-run env PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python experiments/reuse_variable_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D

from commute import figures as FG

HERE = Path(__file__).resolve().parent
PAIRS = {"Always on": "re-weighted", "Cue": "rewired", "XOR": "built", "Table": "rewired"}  # label by construction, second wording
K = 4
SEARCH_BAR = 0.9
OFFSET = {"G": -0.15, "H": 0.15}
METHOD = {"G": "regating", "H": "steering"}  # colours only; the legend below names the series


def found(stage2: dict[str, Any], share: str, pair: str) -> float:
    """Best found value: G over the four sites and three starts at k = 4; H over the five sites."""
    prefix = f"{pair}/k{K}/" if share == "G" else f"{pair}/"
    return max(v[share] for key, v in stage2[share].items() if key.startswith(prefix))


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    stage1 = json.loads((results / "stage1.json").read_text())["pairs"]
    stage2 = json.loads((results / "stage2.json").read_text())
    fig, ax = FG.figure(FG.ONE_COLUMN)
    rows = [f"{p} ({lab})" for p, lab in PAIRS.items()]
    recs: list[dict[str, Any]] = []
    for share in ("G", "H"):
        vals = {row: [found(stage2, share, p)] for row, p in zip(rows, PAIRS)}
        FG.dots(ax, vals, method=METHOD[share], xlabel="share of the gap closed", offset=OFFSET[share])
        color = FG.METHODS[METHOD[share]].color
        for i, p in enumerate(PAIRS):
            w = stage1[p]["witness"][share]
            recs.append({"pair": p, "share": share, "found": vals[rows[i]][0], "witness": w})
            if w is not None:
                ax.scatter([w], [i + OFFSET[share]], s=60, facecolors="none", edgecolors=color, linewidths=1.0)
    FG.reference(ax, SEARCH_BAR, "search bar")
    FG.reference(ax, 0.0, "identity map")
    gc, hc = FG.METHODS["regating"], FG.METHODS["steering"]
    ax.legend(handles=[
        Line2D([], [], color=gc.color, marker=gc.marker, linestyle="", label=f"G found, k = {K}"),
        Line2D([], [], color=hc.color, marker=hc.marker, linestyle="", label="H found"),
        Line2D([], [], color="#555555", marker="o", markerfacecolor="none", linestyle="", label="witness, known by construction"),
    ], fontsize=7, loc="center", frameon=False)
    ax.set_xlim(-0.1, 1.08)
    ax.set_title("Does the blind search reach the known witnesses?", fontsize=9)
    FG.save(fig, out / "found_vs_witness.png", n="best of 3 starts x 4 sites for G, of 5 sites for H; d on 100,000 held-out windows per pair",
            data=recs)


if __name__ == "__main__":
    main()
