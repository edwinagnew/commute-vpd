"""E228's one figure: per pair (rows) the fine-tunes' own KL-form share (filled: reuse side blue, new side red), their
KL-matched random draws (grey dots) and the structured control (black cross), on one axis; r beside it in a second
panel. From results/states/*.json and score.json (no GPU):

    ~/commute-agents/fleet/cpu-run python3 experiments/random_kl_share/plot.py   -> figures/random_share.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
RUN: Any = load_experiment("random_kl_share", "run")
REUSE, NEW = "#2a6fdb", "#d9534f"


def num(x: object) -> float:
    return float(x) if isinstance(x, int | float) else math.nan


def main() -> None:
    pairs, _, _ = RUN.catalogue()
    s = json.loads((RUN.RESULTS / "score.json").read_text())
    recs = {p.stem: json.loads(p.read_text()) for p in RUN.STATES.glob("*.json")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(pairs) + 1.5), sharey=True)
    for ax, (rule, key) in zip(axes, (("K1", "kl_share"), ("P1", "r")), strict=True):
        for y, (p, (reuse, new)) in enumerate(pairs.items()):
            for side, col, dy in ((reuse, REUSE, -0.15), (new, NEW, 0.15)):
                for st in side:
                    rec = recs.get(RUN.fname(st))
                    if rec is None:
                        continue
                    ax.scatter(num(rec["own"][key]), y + dy, color=col, s=18, zorder=3)
                    ax.scatter([num(d.get(key)) for d in rec["draws"].values()], [y + dy] * len(rec["draws"]), color="#999999", s=8, zorder=2)
                    ax.scatter(num(rec["control"].get(key)), y + dy, color="k", marker="x", s=14, zorder=4)
        ax.set_yticks(range(len(pairs)), list(pairs), fontsize=7)
        r = s["rules"][rule]
        ax.set_title(f"{rule}, {key}: {r['overall']} (band {s['bands'][key]['band']:.3f})", fontsize=9)
        ax.set_xlabel(key)
    axes[0].invert_yaxis()
    fig.suptitle(f"{s['experiment']}: fine-tunes (blue reuse side, red new side), KL-matched random draws (grey), structured control (x)", fontsize=9)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "random_share.png", dpi=130)


if __name__ == "__main__":
    main()
