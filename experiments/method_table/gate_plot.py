"""P2.22, the part 2 gate figure: every 67M task pair by every method, each cell the side built to reuse minus the side built to be new.

Saved results only, no model. Reads `results/final.json` (nine methods, P2.11 to P2.21), E155's top-neuron overlap
(`experiments/neuron_overlap/results/stage2.json`, P2.18) and part 1's re-weighting share gap on the five pairs both parts
read (`experiments/part1_vs_methods/results/compare.json`, P2.20). The last row is the pair P2.17 tried to build and could
not, drawn as not run. Writes `figures/gate_scorecard.png`, `.pdf` and `.csv`. Labels are as each pair was built; no
verdict of reused or new, and no combined score. A boxed cell is one whose two sides are apart by the cell's own rule.

    ~/commute-agents/fleet/cpu-run experiments/method_table/gate_plot.py
"""

import json
from pathlib import Path
from typing import Any

from matplotlib.patches import Rectangle

from commute.figures import figure, gap_heatmap, save

ROOT = Path(__file__).resolve().parents[2]
FINAL, OVERLAP, PART1 = "experiments/method_table/results/final.json", "experiments/neuron_overlap/results/stage2.json", "experiments/part1_vs_methods/results/compare.json"
COLS = {"share": "re-gating\nshare", "untrained": "share on\nuntrained\nSVD", "neuron": "neuron,\nwhole\nwidth", "neuron_half": "neuron,\nhalf",
        "neuron_top": "neuron,\ntop sets", "overlap": "top-240\noverlap", "on_both": "change on\nneurons\nin both", "steering": "steering",
        "prior": "prior\n(log10)", "size": "update\nsize\n(log10)", "erank": "effective\nrank\n(log10)", "part1": "part 1\nre-weighting\nshare"}  # fmt: skip
NOT_BUILT = "Prior and steering opposed: not built"
FOOTER = (
    "67M model. Each cell: side built to reuse minus side built to be new; shares and log10 prior a difference of seed means, "
    "size and rank log10 of the ratio. Colour capped at 1. Box = sides apart by the cell's rule (seed ranges, a run's filed call, "
    "or for the overlap columns the wider side's seed range). x = not run. Seeds per side 1 to 8; part 1 two seeds a side. "
    "Neuron cells on the two label twins were read on 2026-11-11."
)
Cell = dict[str, Any]


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def cells() -> tuple[dict[tuple[str, str], Cell], list[str], dict[str, str], set[str]]:
    final = load(FINAL)["cells"]
    out: dict[tuple[str, str], Cell] = {(c["pair"], c["method"]): {"gap": c["gap"], "apart": c["call"] == "apart", "source": f"{FINAL} {c['source']}"} for c in final}
    rows = list(dict.fromkeys(c["pair"] for c in final))
    names = {c["pair"]: c["pair_name"] for c in final}
    built = {c["pair"] for c in final if c["group"] == "built"}
    for pair, s in load(OVERLAP)["score"].items():
        for col in ("overlap", "on_both"):
            out[(pair, col)] = {"gap": s[col]["gap"], "apart": s[col]["call"] != "no call", "source": f"{OVERLAP} .score.{pair}.{col}.gap"}
    for key, p in load(PART1)["part1"].items():
        pair = load(PART1)["overlapping"][key]["final_pair"]
        out[(pair, "part1")] = {"gap": p["gap"], "apart": False, "source": f"{PART1} .part1.{key}.gap"}
    return out, rows, names, built


def main() -> None:
    table, rows, names, built = cells()
    labels = [names[p] for p in rows] + [NOT_BUILT]
    values = [[table[(p, c)]["gap"] if (p, c) in table else None for c in COLS] for p in rows] + [[None] * len(COLS)]
    fig, ax = figure((11.0, 7.4))
    gap_heatmap(ax, values, labels, list(COLS.values()), label="reuse side minus new side (colour capped at 1)", vmax=1.0)
    for i, p in enumerate(rows):
        for j, c in enumerate(COLS):
            if table.get((p, c), {}).get("apart"):
                ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, edgecolor="black", linewidth=1.1))
    n_built = sum(p in built for p in rows)
    for y in (n_built - 0.5, len(rows) - 0.5):
        ax.axhline(y, color="black", linewidth=1.5)
    ax.axvline(len(COLS) - 1.5, color="black", linewidth=1.0, linestyle="--")
    ax.xaxis.tick_top()
    ax.set_title("Only the re-gating share puts the reuse side apart and higher on all nine built pairs", fontsize=11, pad=8)
    data = [{"pair": lab, "method": c.replace("\n", " "), "gap": v, "apart": table.get((p, c), {}).get("apart"), "source": table.get((p, c), {}).get("source")}
            for p, lab, row in zip(rows + [""], labels, values, strict=True) for c, v in zip(COLS, row, strict=True)]  # fmt: skip
    save(fig, ROOT / "experiments/method_table/figures/gate_scorecard.png", n=FOOTER, data=data)


if __name__ == "__main__":
    main()
