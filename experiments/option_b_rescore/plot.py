"""P1.49 figure from results/summary.json only: per built item, R under wording A, the search-based R_b (E247's
1 - c_kept) and the order-free E_b (descriptive), on one axis.

    ~/commute-agents/fleet/cpu-run python3 experiments/option_b_rescore/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
ORDER = ("do_nothing", "x2", "x10", "rebuilt", "new_few", "new_thin", "random_size_241001", "random_kl_241002")
SHORT = {"do_nothing": "do-nothing", "x2": "gain x2", "x10": "gain x10", "rebuilt": "rebuilt", "new_few": "new, few parts",
         "new_thin": "new, thin", "random_size_241001": "random\n(size)", "random_kl_241002": "random\n(KL)"}  # fmt: skip
SERIES = (("R_A", "R under wording A, counts parts", "spectrum"), ("R_b", "R_b: change carried by stood-in parts, E242's search", "neuron"),
          ("E_b", "change M's mechanism explains, each part on its own (descriptive)", "regating"))  # fmt: skip


def main() -> None:
    items = json.loads((HERE / "results" / "summary.json").read_text())["items"]
    rows = [{"item": i, "R_A": items[i]["R_A"], "R_b": items[i]["R_b"]["bar 0.1 (filed)"], "E_b": items[i]["E_b"]} for i in ORDER]
    fig, ax = FG.figure(FG.FULL_WIDTH)
    for k, (key, label, method) in enumerate(SERIES):
        st = FG.METHODS[method]
        ax.scatter([j + (k - 1) * 0.18 for j in range(len(rows))], [r[key] for r in rows], color=st.color, marker=st.marker, s=30, label=label)
    ax.axvline(0.5, color="#BBBBBB", linewidth=0.8)
    ax.axvline(3.5, color="#BBBBBB", linewidth=0.8)
    ax.axvline(5.5, color="#BBBBBB", linewidth=0.8)
    ax.set_xticks(range(len(rows)), [SHORT[r["item"]] for r in rows], fontsize=7)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("share reused, 1 = all of the change")
    ax.set_title("Weighted by change, the search-based reading ranks both gain edits below random", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.46, 0.8))
    FG.save(fig, HERE / "figures" / "r_b_built.png", n="one reading per item on all of D, VPD featurization, 67M", data=rows)


if __name__ == "__main__":
    main()
