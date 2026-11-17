"""E135's one figure, from results/stage1.json only (no model): eight rows of dots, one per method and side (the share, the
pooled and per-key steering vectors at their tuned settings, and matched random vectors), three seeds each, in log-odds
shares of the fine-tune's gain on the 100 score windows. Log P columns stay in the JSON.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/steering_copy_lookup/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SIDES, SEEDS = ("copy", "lookup"), (0, 1, 2)


def rows(S: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per method, side and seed: the plotted numbers, as `figures.save` writes them to the CSV."""
    methods = {"share": ("regating", lambda s: s["r_score"]), "steering, pooled": ("steering", lambda s: s["steering"]["pooled"]["share_lo"]),
               "steering, per key": ("steering", lambda s: s["steering"]["perkey"]["share_lo"]), "random vector": ("random", lambda s: s["steering"]["pooled"]["random"]["mean_lo"])}  # fmt: skip
    return [{"method": name, "style": style, "side": side, "seed": seed, "value": float(f(S[f"{side}_s{seed}"]))}
            for name, (style, f) in methods.items() for side in SIDES for seed in SEEDS]  # fmt: skip


def main(results: Path = RESULTS / "stage1.json", out: Path = HERE / "figures" / "steering_copy_lookup.png") -> None:
    data = json.loads(results.read_text())
    recs = rows(data["states"])
    labels = list(dict.fromkeys(f"{r['method']}, {r['side']}" for r in recs))
    fig, ax = FG.figure(FG.FULL_WIDTH)
    for method, style in dict.fromkeys((r["method"], r["style"]) for r in recs):
        groups: dict[str, list[float]] = {lab: [] for lab in labels}
        for r in recs:
            if r["method"] == method:
                groups[f"{method}, {r['side']}"].append(r["value"])
        FG.dots(ax, groups, method=style, xlabel="share of the fine-tune's gain (log-odds), 100 score windows")
    FG.reference(ax, 0.0, "no share")
    score = data["score"]
    ax.set_title(f"The share's copy-minus-lookup gap is {float(score['gap_share']):.2f}; steering's, the wider of its two kinds, {float(score['gap_steering']):.2f}", fontsize=9)
    FG.save(fig, out, n="3 fine-tune seeds per side, 100 score windows; random vector: mean of 8 draws per seed (pooled kind)", data=recs)


if __name__ == "__main__":
    main()
