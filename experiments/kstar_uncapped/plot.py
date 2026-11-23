"""E269 (cell P2.61) figure from results/summary.json (score.py).

figures/eb_capped_uncapped.png: E_b of every sequence-task seed (SFT, RL, guarded RL) and the matched random update,
read over K* from the search capped at 2,000 evaluations (E257) beside K* from the uncapped search, on one axis.
Writes .png, .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/kstar_uncapped/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.lines import Line2D

from commute import figures as FG

HERE = Path(__file__).resolve().parent
KINDS = {"sft": "SFT", "rl": "RL", "rlg": "RL, guarded", "random": "random update"}
SEARCHES = {"E_b_capped": ("neuron", -0.15, "search capped at 2,000 evaluations"),
            "E_b": ("regating", 0.15, "search run to the end, no cap")}  # fmt: skip
GAP_BAR = 0.01  # the filed bar on a kind's mean minus SFT's (results/predictions.json, rules)


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    rows = [(item.split("sequence_")[1].rsplit("_s", 1)[0], v) for item, v in s["items"].items()]
    fig, ax = FG.figure(FG.FULL_WIDTH)
    for key, (method, off, _) in SEARCHES.items():
        groups = {name: [v[key] for kind, v in rows if kind == k] for k, name in KINDS.items()}
        FG.dots(ax, groups, method=method, xlabel="E_b on the sequence task, each seed", offset=off)
    sft = sum(v["E_b"] for kind, v in rows if kind == "sft") / 3
    bar = GAP_BAR
    ax.axvspan(sft, sft + bar, color="#0072B2", alpha=0.12, linewidth=0)
    ax.annotate(f"uncapped SFT mean {sft:.3f}\nto {bar} above it", (sft + bar / 2, 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=7, color="#0072B2")  # fmt: skip
    handles = [Line2D([], [], linestyle="none", marker=FG.METHODS[m].marker, color=FG.METHODS[m].color, label=label)
               for m, _, label in SEARCHES.values()]  # fmt: skip
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    ax.set_xlim(0, 0.06)
    ax.set_title("Lifting the search cap raises E_b a little for every state;\nRL and SFT still read alike")
    data = [{"item": v["item"], "kind": kind, "E_b_capped": v["E_b_capped"], "E_b_uncapped": v["E_b"],
             "size_K_capped": v["size_K_capped"], "size_K_uncapped": v["size_K"]} for kind, v in rows]  # fmt: skip
    FG.save(fig, HERE / "figures" / "eb_capped_uncapped.png",
            n="3 training seeds per kind, 1 random update; capped values from the earlier run's searches;\n"
              "shaded: where a kind's mean would sit if it were within the filed 0.01 of SFT's", data=data)


if __name__ == "__main__":
    main()
