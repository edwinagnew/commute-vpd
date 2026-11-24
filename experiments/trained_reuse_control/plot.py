"""E297's figure (cell P1.4 run 2), from results/summary.json and results/items/{x2,new_few}.json only (no model): the
write share E_b over K per item at neurons (all three seeds) and at VPD (seed 0), for the fine-tunes trained to copy
x2 and new_few, their size-matched random updates, and the two built edits themselves as references. The x2 copies
missed the filed closeness bar on every seed, so the figure is descriptive: no rival call is drawn.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/trained_reuse_control/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
ROWS = {  # row label -> (kind, target); kind "built" reads the built edit, "ft" the copies, "rand" their randoms
    "built x2 (reuse)": ("built", "x2"),
    "fine-tunes copying x2\n(not close to x2)": ("ft", "x2"),
    "their matched randoms": ("rand", "x2"),
    "built new_few (new)": ("built", "new_few"),
    "fine-tunes copying new_few\n(close)": ("ft", "new_few"),
    "their matched randoms ": ("rand", "new_few"),
}
XLABEL = "E_b over K, log scale"


def records(summary: dict[str, Any], results: Path) -> list[dict[str, Any]]:
    """One record per plotted point: featurization, row, seed, E_b over K, |K|, median margin."""
    out: list[dict[str, Any]] = []
    for world, table, seeds in (("neurons", summary["neurons"], SEEDS), ("vpd", summary["vpd"], (0,))):
        for row, (kind, target) in ROWS.items():
            for s in seeds if kind != "built" else (None,):
                if kind == "built":
                    path = results / ("items" if world == "neurons" else "vpd") / f"{target}.json"
                    rec = json.loads(path.read_text())
                else:
                    rec = table[f"ft/{target}_s{s}" + ("_rand" if kind == "rand" else "")]
                out.append({"world": world, "row": row.replace("\n", " ").strip(), "kind": kind, "target": target,
                            "seed": s, "E_b_K": rec["E_b_K"], "size_K": rec["size_K"],
                            "margin_median": rec["margin_median"]})
    return out


def panel(ax: Any, recs: list[dict[str, Any]], world: str, title: str) -> None:
    def values(want: set[str]) -> dict[str, list[float]]:
        return {row: [r["E_b_K"] for r in recs if r["world"] == world and (r["kind"], r["target"]) == ROWS[row]]
                if ROWS[row][0] in want else [] for row in ROWS}

    FG.dots(ax, values({"built", "ft"}), method="neuron", xlabel=XLABEL)
    FG.dots(ax, values({"rand"}), method="random", xlabel=XLABEL)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 2.0)
    FG.reference(ax, 1.0, "do-nothing 1")
    ax.axhline(2.5, color="#cccccc", linewidth=0.6, linestyle="-")
    ax.set_title(title, fontsize=8, loc="left", pad=12)


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    summary = json.loads((results / "summary.json").read_text())
    assert summary["R3"]["call"] == "no call", summary["R3"]
    recs = records(summary, results)
    fig, (a, b) = FG.figure(FG.FULL_WIDTH, ncols=2)
    panel(a, recs, "neurons", "neurons, seeds 0-2")
    panel(b, recs, "vpd", "VPD subcomponents, seed 0")
    b.set_yticklabels([])
    fig.suptitle("Copies of x2 read like copies of new_few, far below x2 itself;\n"
                 "but the x2 copies never got close to x2, so this is no test", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FG.save(fig, out / "e_b_imitators.png", n="3 seeds per fine-tune row at neurons, 1 at VPD; x2 rows on its 75 held-out "
            "windows at 41 positions, new_few rows on 512 cue windows at the last position", data=recs)


if __name__ == "__main__":
    main()
