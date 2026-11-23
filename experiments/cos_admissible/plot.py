"""E### (cell P1.55) figure from results/summary.json: figures/cos2_vs_eb.png.

Left: the one-gain squared-cosine share over K against the write share E_b over K, one dot per saved state, coloured
by E268's groups, with y = x. Right: the same two readings as shortfalls from their top, 1 - reading, on log axes with
y = x: per variable 1 - cos2 = (1 - share) dd / |F|^2, so the dots sit below the line by about the relative change
size. The state with K empty reads 1 on both (convention) and is left out of the right panel. Writes .png, .pdf, .csv.

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


def rows_of(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"item": r["item"], "group": E268P.group(r["item"]), "cos2_K": r["cos2_K"], "E_b_K": r["E_b_K"],
             "K_empty": r["n_changed_K"] == 0, "rel_change_median": r["rel_change_median"], "top": r["top"], "R_eq_K": r["R_eq_K"]}
            for r in summary["items"].values()]  # fmt: skip


def panel(ax: Any, rows: list[dict[str, Any]], short: bool) -> None:
    f = (lambda v: 1 - v) if short else (lambda v: v)
    shown = [r for r in rows if not (short and r["K_empty"])]
    lo = 1e-12 if short else 0.0
    grid = np.geomspace(1e-12, 1, 200) if short else np.linspace(0, 1, 2)
    ax.plot(grid, grid, color="#555555", linewidth=0.8, linestyle="--", label="equal: y = x")
    for name, method in E268P.GROUPS.items():
        st, pts = FG.METHODS[method], [r for r in shown if r["group"] == name]
        if pts:
            ax.scatter([max(f(r["E_b_K"]), lo) for r in pts], [max(f(r["cos2_K"]), lo) for r in pts], color=st.color, marker=st.marker,
                       s=24, alpha=0.85, linewidths=0, label=f"{name} ({len(pts)})")  # fmt: skip
    if short:
        ax.set(xscale="log", yscale="log", xlim=(1e-3, 1.2), ylim=(1e-12, 1.2))
        ax.set_xlabel("1 - write share over K")
        ax.set_ylabel("1 - squared-cosine share over K")
        ax.set_title("Shortfalls from the top: the squared cosine\nsits below the write share by the change size", fontsize=9)
    else:
        ax.set(xlim=(0, 1.02), ylim=(0, 1.02))
        ax.set_xlabel("write share over K (E268)")
        ax.set_ylabel("squared-cosine share over K\n(one gain, sign checked, F's settings)")
        ax.set_title("Squared-cosine share against the write share,\nboth over the needed set", fontsize=9)
        ax.legend(fontsize=6, loc="lower right", frameon=False)


def main(summary: Path, out: Path) -> None:
    rows = rows_of(json.loads(summary.read_text()))
    fig, (a, b) = FG.figure((10.0, 4.8), ncols=2)
    panel(a, rows, short=False)
    panel(b, rows, short=True)
    n_empty = sum(r["K_empty"] for r in rows)
    FG.save(fig, out, data=rows,
            n=f"{len(rows)} saved states, one dot each; K from 256 pairs of inputs per state (E268). The right panel leaves out "
              f"the {n_empty} state(s) with no changed needed variable (1 on both by convention); shortfalls below 1e-12 drawn at 1e-12.")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json",
         Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures" / "cos2_vs_eb.png")  # fmt: skip
