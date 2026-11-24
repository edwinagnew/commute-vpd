"""E274 (cell P1.55) figure from results/summary.json: figures/cos2_vs_eb.png.

Left: the one-gain squared-cosine share over K against the write share E_b over K, one dot per saved state, coloured
by E268's groups, with y = x. Right: the squared cosine's shortfall from its top, 1 - cos2 over K, against the median
change size dd / |F|^2 over the same variables, log axes, with y = x: per variable 1 - cos2 = (1 - share) dd / |F|^2,
and the share of a single variable is mostly near 0, so the shortfall follows the change size. The state with no
changed needed variable reads 1 on both (convention) and is left out of the right panel. Writes .png, .pdf, .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/cos_admissible/plot.py [SUMMARY] [OUT_PNG]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from commute import figures as FG
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
E268P: Any = load_experiment("kstar_needed", "plot")  # E268's state groups and their styles
NAMED = {"x2": "gain x2", "x10": "gain x10", "rebuilt": "rebuilt", "new_thin": "new (thin)"}


def rows_of(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"item": r["item"], "group": E268P.group(r["item"]), "cos2_K": r["cos2_K"], "E_b_K": r["E_b_K"],
             "K_empty": r["n_changed_K"] == 0, "rel_change_median": r["rel_change_median"], "top": r["top"], "R_eq_K": r["R_eq_K"]}
            for r in summary["items"].values()]  # fmt: skip


def dots(ax: Any, rows: list[dict[str, Any]], x: str, y: Any) -> None:
    for name, method in E268P.GROUPS.items():
        st, pts = FG.METHODS[method], [r for r in rows if r["group"] == name]
        if pts:
            ax.scatter([r[x] for r in pts], [y(r) for r in pts], color=st.color, marker=st.marker, s=24, alpha=0.85, linewidths=0,
                       label=f"{name} ({len(pts)})")  # fmt: skip
    for r in rows:
        if r["item"] in NAMED:
            ax.annotate(NAMED[r["item"]], (r[x], y(r)), xytext=(5, -3), textcoords="offset points", fontsize=7)


def left(ax: Any, rows: list[dict[str, Any]]) -> None:
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.8, linestyle="--", label="equal: y = x")
    dots(ax, rows, "E_b_K", lambda r: r["cos2_K"])
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02))
    ax.set_xlabel("write share over K (E268)")
    ax.set_ylabel("squared-cosine share over K\n(one gain, sign checked, F's settings)")
    ax.set_title("The squared cosine sits near 1 on every state but x10;\nit does not order the states as the write share does", fontsize=9)
    ax.legend(fontsize=6, loc="lower center", frameon=False)


def right(ax: Any, rows: list[dict[str, Any]]) -> None:
    shown = [r for r in rows if not r["K_empty"]]
    grid = np.geomspace(1e-5, 1, 2)
    ax.plot(grid, grid, color="#555555", linewidth=0.8, linestyle="--", label="equal: y = x")
    dots(ax, shown, "rel_change_median", lambda r: 1 - r["cos2_K"])
    ax.set(xscale="log", yscale="log", xlim=(1e-5, 1.2), ylim=(1e-5, 1.2))
    ax.set_xlabel("median change size over K, |F - M|^2 / |F|^2 per variable")
    ax.set_ylabel("1 - squared-cosine share over K")
    ax.set_title("Its shortfall from 1 follows the size of the change,\nnot how much of the change is a gain", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", frameon=False)


def main(summary: Path, out: Path) -> None:
    rows = rows_of(json.loads(summary.read_text()))
    fig, (a, b) = FG.figure((10.0, 4.8), ncols=2)
    left(a, rows)
    right(b, rows)
    n_empty = sum(r["K_empty"] for r in rows)
    FG.save(fig, out, data=rows,
            n=f"{len(rows)} saved states, one dot each; K from 256 pairs of inputs per state (E268). The right panel leaves out "
              f"the {n_empty} state with no changed needed variable (1 on both by convention).")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json",
         Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures" / "cos2_vs_eb.png")  # fmt: skip
