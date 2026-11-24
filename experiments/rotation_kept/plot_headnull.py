"""E295 (cell P1.60 run 2) figure, CPU only, from results/headnull/ (no model).

Draws figures/headnull_kept.png: head (2,4)'s kept share on all of D for each copy and lookup fine-tune beside its
size-matched random update, with the pure-strengthening control (head 4's value rows doubled, v2) and the do-nothing,
both reading 1, marked.

    python experiments/rotation_kept/plot_headnull.py
"""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.lines import Line2D

from commute import figures as FG

HERE = Path(__file__).parent
RES = HERE / "results" / "headnull"
BASES = [f"{arm}_s{s}" for arm in ("copy", "lookup") for s in range(3)]


def main() -> None:
    def kept(name: str) -> float:
        return float(json.loads((RES / f"e62__{name}.json").read_text())["whole_d"]["kept"])

    for ref in ("v2", "do_nothing"):
        assert json.loads((RES / f"{ref}.json").read_text())["whole_d"]["kept"] > 1 - 1e-9, ref
    rows = [{"item": b, "fine_tune": kept(b), "matched_random": kept(f"{b}_rand")} for b in BASES]
    labels = [f"{b.split('_')[0]}, seed {b[-1]}" for b in BASES]
    fig, ax = FG.figure((5.6, 3.6))
    for key, method, off in (("fine_tune", "regating", -0.15), ("matched_random", "random", 0.15)):
        FG.dots(ax, {lab: [r[key]] for lab, r in zip(labels, rows)}, method=method,
                xlabel="head (2,4)'s kept share on all of D, log scale", offset=off)  # fmt: skip
    FG.reference(ax, 1.0, "1: pure strengthening\nand do-nothing")
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 3.0)
    ax.axhline(2.5, color="#bbbbbb", linewidth=0.6)
    handles = [Line2D([], [], color=FG.METHODS[m].color, marker=FG.METHODS[m].marker, linestyle="", markersize=5, label=t)
               for m, t in (("regating", "fine-tune"), ("random", "random update, same size per tensor"))]  # fmt: skip
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, 1.0), fontsize=7, frameon=False)
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Head (2,4) keeps far more of its write under the copy and lookup fine-tunes\n"
                 "than under a random update of the same size, and far less than a pure strengthening", fontsize=9)  # fmt: skip
    FG.save(fig, HERE / "figures" / "headnull_kept.png",
            n="one point per state: 3 seeds per arm, one matched random each; 512 windows of D, last position",
            data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
