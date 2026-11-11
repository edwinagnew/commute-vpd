"""Cell P2.20's one figure: per method, its gap on each overlapping pair against part 1's gap on the same pair, one panel
per method on a shared x axis. Filled marker: the method's call is 'apart'; open: no call or one value a side; a pair
with no reading is absent. Reads results/compare.json only.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/part1_vs_methods/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
LABEL = {"share": "re-gating share", "steering": "steering share", "prior": "prior, log10", "erank": "effective rank, log10 ratio",
         "size": "update size, log10 ratio", "untrained": "share on untrained SVD", "neuron": "neuron, whole width",
         "neuron_half": "neuron, half", "neuron_top": "neuron, top neurons"}  # fmt: skip
STYLE = {"share": "regating", "steering": "steering", "prior": "prior", "erank": "spectrum", "size": "size", "neuron": "neuron",
         "neuron_half": "neuron_half", "neuron_top": "random", "untrained": "random"}  # fmt: skip
SHORT = {"e62_copy": "copy|lookup", "copy24": "copy|cipher", "ident32": "ident|rand", "case_real": "case", "repeat_real": "repeat"}


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    res = json.loads((results / "compare.json").read_text())
    p1 = {a: res["part1"][a]["gap"] for a in res["overlapping"]}
    fig, axes = FG.figure((7.5, 6.6), ncols=3, nrows=3)
    rows = []
    for ax, (m, r) in zip(axes.flat, res["methods"].items(), strict=True):
        style = FG.METHODS[STYLE[m]]
        for a, c in r["pairs"].items():
            if c["gap"] is None:
                continue
            apart = c["call"] == "apart"
            ax.scatter(p1[a], c["gap"], s=30, marker=style.marker, color=style.color if apart else "none", edgecolors=style.color, linewidths=1)
            ax.annotate(SHORT[a], (p1[a], c["gap"]), xytext=(3, 2), textcoords="offset points", fontsize=6, color="#555555")
            rows.append({"method": m, "pair": a, "part1_gap": p1[a], "method_gap": c["gap"], "call": c["call"], "order": c["order"]})
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.annotate("no difference", (0.0, 0), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points",
                    fontsize=6, color="#555555", ha="left")  # fmt: skip
        k = r["counts"]
        ax.set_title(f"{LABEL[m]}\n{k['agree']} agree, {k['reverse']} reverse, {k['no order']} no order, {k['missing']} missing", fontsize=7)
        ax.set_ylabel("method gap", fontsize=7)
        ax.set_xlim(0.1, 0.5)
        ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("part 1 gap in G, reuse minus new", fontsize=7)
    fig.suptitle("Steering ranks the five shared pairs as part 1 does; no method agrees with part 1 in sign on all five", fontsize=8)
    FG.save(fig, out / "part1_vs_methods.png", data=rows,
            n="One point per pair, both seeds of part 1 averaged; each method's gap is reuse side minus new side over its own seeds, 1 to 8 per side. "
              "Filled: the method's sides are apart; open: no call or one value a side.")  # fmt: skip


if __name__ == "__main__":
    main()
