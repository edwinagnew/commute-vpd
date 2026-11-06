"""E117's figure, figures/panels.png, from results/cells.json and score.json and the two full-width references, no model.
Four panels on one row: A (seed agreement of the set), B (restore share of the set), C (re-switching carried by the set),
and the full-width reference strip (VPD's r; neurons' r_n filled, r_half hollow). In each panel the two families
(copy vs random maps; lowercase vs the two E53 ciphers) each show VPD then neurons, one point per state (A: per task).
Built to reuse (existing) is blue, built to be new is orange (squares: the broad cipher); the grey bar behind a point
is the range of its 20 random same-count sets. The caption gives the thresholds and each panel's verdicts.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_vpd_count/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
RES, FIG, EXP = HERE / "results", HERE / "figures", HERE.parent
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#c9c8c3", "#52514e", "#e4e3df"
FAMILIES = {"copy vs random": (["ident32", "ident32b"], ["rand32", "rand32b"]),
            "lowercase vs ciphers": (["lowercase"], ["e53_cipher", "broad_cipher"])}
METHODS = ("vpd", "neurons")


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def style(task: str, existing: list[str]) -> dict[str, Any]:
    return {"color": BLUE if task in existing else ORANGE, "marker": "s" if task == "broad_cipher" else "o"}


def slots() -> list[tuple[str, str, float]]:
    """(family, method, x) for the four groups of every panel."""
    return [(f, m, 2.5 * i + j) for i, f in enumerate(FAMILIES) for j, m in enumerate(METHODS)]


def draw(ax: Axes, get: Any, title: str) -> None:
    """`get(family, method, task, seed)` -> (value or None, (lo, hi) or None); seed None for panel A."""
    for fam, method, x in slots():
        ex, nw = FAMILIES[fam]
        for i, task in enumerate(ex + nw):
            for s in (None,) if title.startswith("A") else (0, 1):
                v, band = get(fam, method, task, s)
                dx = x + 0.12 * (i - (len(ex + nw) - 1) / 2) + (0 if s is None else 0.04 * (2 * s - 1))
                if band is not None:
                    ax.plot([dx, dx], band, color=GREY, lw=4, solid_capstyle="round", zorder=1)
                if v is not None:
                    ax.scatter([dx], [v], s=36, zorder=3, edgecolor="white", linewidth=1, **style(task, ex))
    ax.set_xticks([x for _, _, x in slots()], [f"{m}\n{f}" if m == "vpd" else m for f, m, _ in slots()], fontsize=7)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> None:
    cells, score = load(RES / "cells.json"), load(RES / "score.json")["verdicts"]
    rows, jac = cells["rows"], cells["jaccard"]
    table, e98 = load(EXP / "regating_truth_table" / "results" / "table.json")["runs"], load(EXP / "neuron_r" / "results" / "cells.json")["rows"]

    def shares(panel: str) -> Any:
        def get(_f: str, method: str, task: str, s: int | None) -> tuple[float | None, tuple[float, float] | None]:
            b = rows[f"{task}_s{s}"][panel][method]
            r = b["random"]
            return (b["set"] if b["readable"] else None), ((r["min"], r["max"]) if r and b["readable"] else None)

        return get

    def a(_f: str, method: str, task: str, _s: int | None) -> tuple[float | None, None]:
        j = jac.get(task)
        return (j[method] if j and j["readable"] else None), None

    def ref(_f: str, method: str, task: str, s: int | None) -> tuple[float | None, tuple[float, float] | None]:
        n = f"{task}_s{s}"
        if method == "vpd":
            pre = "e46/" if task in FAMILIES["copy vs random"][0] + FAMILIES["copy vs random"][1] else "e53/"
            return table[pre + n.replace("e53_cipher", "cipher")]["r"]["value"], None
        return e98[n]["r_n"], (e98[n]["r_half"], e98[n]["r_half"] + 1e-3)  # r_half drawn as a short grey mark

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=False)
    draw(axes[0], a, "A  seed agreement of the set (Jaccard, per task)")
    draw(axes[1], shares("B"), "B  restore share of the set")
    draw(axes[2], shares("C"), "C  re-switching carried by the set")
    draw(axes[3], ref, "ref  full width, not rerun (VPD r; neurons r_n, grey r_half)")
    axes[0].set_ylabel("share of the gain, or Jaccard", fontsize=8, color=INK)
    v = lambda k: score[k]["verdict"]
    lines = [f"{p}: VPD {v(f'copy_vs_random/{p}/vpd')}, neurons {v(f'copy_vs_random/{p}/neurons')} (copy vs random)"
             + ("" if p == "A" else f"; E53 bar VPD {v(f'e53_bar/{p}/vpd')}, neurons {v(f'e53_bar/{p}/neurons')}") for p in "ABC"]
    fig.text(0.01, 0.01, "S_on: alive subcomponents whose mean importance change at the read position is above 0.3; N_on: the same count of "
             "neurons with the largest firing increase. Separation (E12): the two classes' ranges do not overlap and the gap exceeds the larger "
             "within-class range, existing class higher. E53's bar on B and C: lowest lowercase minus highest cipher at least 0.3.\n"
             + " | ".join(lines), fontsize=7, color=INK, wrap=True, va="bottom")
    fig.subplots_adjust(bottom=0.27, wspace=0.25, left=0.05, right=0.99)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "panels.png", dpi=150)
    print(f"wrote {FIG / 'panels.png'}")


if __name__ == "__main__":
    main()
