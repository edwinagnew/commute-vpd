"""Figure for E48 Amendment 1: how many subcomponents carry the E42 copies' gain at the cue?
Per copy, the share of the summed positive revert-one drops carried by the k largest drops, against k. Only the top 20
drops, n50 and n90 were saved, so the curve is exact for k = 1..20 and at n50 and n90, dashed between. References: the
same share if the drops were spread evenly over all alive subcomponents (k / n_alive), and the filed classes' bounds.

Reads results/carriers.json only. Usage (repo root): uv run python experiments/pass_reread/plot_amend1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
# (key, label, colour, label offsets at n50 and at n90 in points, horizontal alignment of each)
COPIES = (("e42/rand32/A", "32 random pairs (filed: tens)", "#c0392b", ((8, -4), (8, -4)), ("left", "left")),
          ("e42/rand1/A", "1 random pair (filed: tens)", "#7d3c98", ((0, -15), (-6, 6)), ("center", "right")),
          ("e42/ident32/A", "identity control, 32 copy pairs", "#eb6834", ((-6, 6), (0, -15)), ("right", "center")))
INK, MUTED = "#0b0b0b", "#8a8984"


def main() -> None:
    rec = json.loads((RESULTS / "carriers.json").read_text())["carriers"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(10, 100, color="#f2f1ec", zorder=0)
    ax.text(31, 0.5,"'tens'\n(filed for both\nrandom ciphers)", ha="center", va="top", fontsize=7.5, color=MUTED)
    for x, text in ((10, "a handful: ≤ 10"), (100, "hundreds: ≥ 100")):
        ax.axvline(x, color=MUTED, ls=":", lw=1)
        ax.text(x * 1.08, 0.28, text, rotation=90, fontsize=7.5, color=MUTED, va="bottom")
    n_alive = next(iter(rec.values()))["n_alive"]
    k = np.logspace(0, np.log10(n_alive), 200)
    ax.plot(k, k / n_alive, color=INK, ls="--", lw=1, label=f"spread evenly over all {n_alive:,} alive")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.text(1.1, 0.51, "half of the summed drops (n50)", fontsize=7.5, color=MUTED, va="bottom")
    for key, label, colour, offsets, aligns in COPIES:
        r = rec[key]
        top = np.cumsum([t["drop"] for t in r["top"]]) / r["pos_sum"]
        ks = np.arange(1, len(top) + 1)
        ax.plot(ks, top, color=colour, lw=2, label=label)
        ax.plot([ks[-1], r["n50"], r["n90"]], [top[-1], 0.5, 0.9], color=colour, ls="--", lw=1, alpha=0.6)
        ax.scatter([r["n50"], r["n90"]], [0.5, 0.9], color=colour, s=45, zorder=3, edgecolor="white")
        for n, y, off, ha in ((r["n50"], 0.5, offsets[0], aligns[0]), (r["n90"], 0.9, offsets[1], aligns[1])):
            ax.annotate(f"{n:,}", (n, y), xytext=off, textcoords="offset points", fontsize=7.5, color=colour, ha=ha)
    ax.plot([], [], color=MUTED, ls="--", lw=1, label="between the saved top 20 and n50, n90: not saved")
    ax.set_xscale("log")
    ax.set_xlim(1, n_alive * 1.1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of subcomponents, largest drop first", fontsize=9)
    ax.set_ylabel("share of the summed drops in the cue's log-odds", fontsize=9)
    ax.set_title("How many subcomponents carry the copies' gain at the cue?\n"
                 "About a thousand or more for half of it, on all three copies; tens were filed", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "e48_a1_carriers.png", dpi=150)
    print(FIGURES / "e48_a1_carriers.png")


if __name__ == "__main__":
    main()
