"""E227's one figure: per method, the six known-answer items' readings oriented so that up = more reuse, reuse items
blue and new items red, unreadable items as crosses on the axis, the band as a bar above the highest new item. From
results/score.json (no GPU):

    ~/commute-agents/fleet/cpu-run python3 experiments/known_answer_baselines/plot.py   -> figures/methods.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LABEL = {"share": "re-gating share", "k_half": "-log2 neurons for half the gain", "erank": "effective rank of the change"}
REUSE, NEW = "#2a6fdb", "#d9534f"


def num(x: object) -> float:
    return float(x) if isinstance(x, int | float) else math.nan


def main() -> None:
    s = json.loads((HERE / "results" / "score.json").read_text())
    pred = json.loads((HERE / "results" / "predictions.json").read_text())["settings"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, rec) in zip(axes, s["methods"].items(), strict=True):
        vals = {i: num(v) for i, v in rec["P1"]["values"].items()}
        finite = [v for v in vals.values() if math.isfinite(v)]
        floor = min(finite) if finite else 0.0
        for k, (i, v) in enumerate(vals.items()):
            c = REUSE if i in pred["reuse"] else NEW
            ax.scatter(k, v if math.isfinite(v) else floor, color=c, marker="o" if math.isfinite(v) else "x", s=40)
        new = [vals[i] for i in pred["new"] if math.isfinite(vals[i])]
        if new:
            ax.axhspan(max(new), max(new) + rec["band"]["band"], color=NEW, alpha=0.15, lw=0)
        ax.set_xticks(range(len(vals)), list(vals), rotation=30, fontsize=8)
        ax.set_title(f"{LABEL[m]}\nP1 {rec['P1']['verdict']}, P1b {rec['P1b']['verdict']}", fontsize=9)
    axes[0].set_ylabel("reading (up = more reuse)")
    fig.suptitle(f"{s['experiment']}: does a baseline put every reuse item (blue) above every new item (red)? crosses: unreadable", fontsize=9)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "methods.png", dpi=130)


if __name__ == "__main__":
    main()
