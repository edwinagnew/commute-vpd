"""E234 figure: how far each changed model's Pile FVE falls below the base's, per SAE site (results/items/*.json).

    ~/commute-agents/fleet/cpu-run python experiments/sae_train/plot.py   -> figures/fve_drop_by_site.{png,pdf,csv}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from commute import figures as F

HERE = Path(__file__).resolve().parent
SMALLEST_GAP = 0.02  # the card's smallest_gap (predictions.json readings.fve_drop)
KNOWN = {"new_few", "new_thin", "rebuilt", "moved", "x2", "x10"}  # E214's six known-answer items
GROUPS = {  # group -> (label, colour, marker)
    "e187": ("E187 fine-tune states (68)", "#0072B2", "o"),
    "known": ("built known-answer items (6)", "#D55E00", "s"),
    "do_nothing": ("do-nothing clone", "#000000", "D"),
}


def group(item: str) -> str:
    return "do_nothing" if item == "do_nothing" else "known" if item in KNOWN else "e187"


def main() -> None:
    items = [json.loads(p.read_text()) for p in sorted((HERE / "results/items").glob("*.json"))]
    sites = list(items[0]["fve_drop"])
    rng = np.random.default_rng(0)
    fig, ax = F.figure(F.FULL_WIDTH)
    data, seen = [], set()
    for it in items:
        g = group(it["item"])
        label, color, marker = GROUPS[g]
        ys = np.arange(len(sites)) + (0 if g == "do_nothing" else rng.uniform(-0.25, 0.25, len(sites)))
        xs = [it["fve_drop"][s] for s in sites]
        ax.scatter(xs, ys, color=color, marker=marker, s=10 if g == "e187" else 18, alpha=0.6 if g == "e187" else 0.9,
                   linewidths=0, label=None if g in seen else label, zorder=3 if g != "e187" else 2)  # fmt: skip
        seen.add(g)
        data += [{"item": it["item"], "group": g, "site": s, "pile_fve_drop": x} for s, x in zip(sites, xs)]
    ax.set_xscale("symlog", linthresh=0.01, linscale=1.0)
    ax.set_xlim(-0.01, 30)
    F.reference(ax, SMALLEST_GAP, "0.02: the card's smallest gap")
    x10 = next(it for it in items if it["item"] == "x10")["fve_drop"]["resid_post3_s0"]
    ax.annotate(f"x10 at resid_post3: {x10:.1f}", (x10, sites.index("resid_post3_s0")), xytext=(-95, 16),
                textcoords="offset points", fontsize=7, arrowprops={"arrowstyle": "-", "color": "#777777"})  # fmt: skip
    ax.set_yticks(range(len(sites)), [s.replace("_s0", "").replace("_s1", ", seed 1") for s in sites])
    ax.invert_yaxis()
    ax.set_xlabel("base FVE minus changed model's FVE, held-out Pile rows (symlog; below 0 = better fit)")
    ax.set_title("46 of 75 changed models lower some base SAE's Pile FVE by over 0.02")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    F.save(fig, HERE / "figures/fve_drop_by_site.png", n="75 states x 14 SAEs, 261,632 Pile validation tokens each",
           data=data)  # fmt: skip


if __name__ == "__main__":
    main()
