"""E137's two figures, from results/stage2.json and results/stage1.json only (no model).

  found_G.png     the found G at k = 4: each pair's best site, its three starts, against 1 (the witness on Always on, the search's
                  own positive control) and 0 (the identity map). The card's question: is Always on above Cue, XOR and Table?
  G_by_k.png      the best found G per pair and k = 1, 4, 16, as a grid of numbers (k = 1 and 16 are for this figure only).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/reuse_variable_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
PAIRS = ("Always on", "Cue", "XOR", "Table")
SITES = ("in1", "in2", "in3", "out3")
KS = (1, 4, 16)
STARTS = 3


def starts_at(G: dict[str, Any], pair: str, k: int, site: str) -> list[float]:
    return [G[f"{pair}/k{k}/{site}/s{s}"]["G"] for s in range(STARTS) if f"{pair}/k{k}/{site}/s{s}" in G]


def best_site(G: dict[str, Any], pair: str, k: int) -> str | None:
    sites = [s for s in SITES if starts_at(G, pair, k, s)]
    return max(sites, key=lambda s: max(starts_at(G, pair, k, s))) if sites else None


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    G = json.loads((results / "stage2.json").read_text())["G"]
    fig, ax = FG.figure(FG.ONE_COLUMN)
    groups: dict[str, list[float]] = {}
    recs: list[dict[str, Any]] = []
    for pair in PAIRS:
        site = best_site(G, pair, 4)
        if site is None:
            continue
        name = f"{pair} (best site {site})"
        groups[name] = starts_at(G, pair, 4, site)
        recs += [{"pair": pair, "site": site, "start": i, "G": v} for i, v in enumerate(groups[name])]
    FG.dots(ax, groups, method="regating", xlabel="G found at k = 4, three starts")
    FG.reference(ax, 1.0, "witness")
    FG.reference(ax, 0.0, "identity map")
    ax.set_title("Found G at k = 4, by pair (lower bounds; a miss is never read as built)", fontsize=9)
    FG.save(fig, out / "found_G.png", n="three search starts per pair, one site each; d on 100,000 held-out windows", data=recs)

    fig, ax = FG.figure(FG.ONE_COLUMN)
    grid: list[list[float | None]] = []
    recs2: list[dict[str, Any]] = []
    for pair in PAIRS:
        row: list[float | None] = []
        for k in KS:
            site = best_site(G, pair, k)
            v = max(starts_at(G, pair, k, site)) if site is not None else None
            row.append(v)
            recs2.append({"pair": pair, "k": k, "site": site, "best_G": v})
        grid.append(row)
    FG.gap_heatmap(ax, grid, PAIRS, [f"k = {k}" for k in KS], label="best found G, best site, best of three starts")
    ax.set_title("Found G by the dimension of the read-out", fontsize=9)
    FG.save(fig, out / "G_by_k.png", n="three starts per site, four sites per k; d on 100,000 held-out windows", data=recs2)


if __name__ == "__main__":
    main()
