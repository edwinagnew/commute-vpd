"""Step 1's figures: where the least-squares split puts a planted change, beside where it puts a random one; and the
re-gating share of the nine states by size. New decomposition p-9707d6ff.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_instrument_check/plot_step1.py            (after step 1)
    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_instrument_check/plot_step1.py --sketch   (the card's sketch)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute import figures as FG

HERE = Path(__file__).resolve().parent
PLANTED = FG.Style("planted change", "#0072B2", "o")
ROWS = (("often", "planted"), ("rarely", "planted"), ("often", "on"), ("rarely", "on"))
LABEL = {"planted": "on the planted set", "on": "on subcomponents already on"}
# Made-up numbers for the card: (kind, reading, planted, random) per rival.
SKETCH = {
    "If the split recovers a placement": [("often", "planted", 0.88, 0.01), ("rarely", "planted", 0.55, 0.002),
                                          ("often", "on", 0.93, 0.35), ("rarely", "on", 0.06, 0.35)],
    "If placement is free": [("often", "planted", 0.02, 0.01), ("rarely", "planted", 0.004, 0.002),
                             ("often", "on", 0.38, 0.35), ("rarely", "on", 0.33, 0.35)],
}


def draw(ax: Axes, recs: list[dict[str, Any]], title: str) -> None:
    """One row per (kind, reading): the planted change's share (blue) and the random change's at the same per-matrix
    energy (grey X), joined."""
    rnd = FG.METHODS["random"]
    for i, r in enumerate(recs):
        ax.plot([r["planted"], r["random"]], [i, i], color="#bbbbbb", linewidth=0.8, zorder=1)
        ax.scatter([r["random"]], [i], color=rnd.color, marker=rnd.marker, s=26, zorder=2, label=rnd.label if i == 0 else None)
        ax.scatter([r["planted"]], [i], color=PLANTED.color, marker=PLANTED.marker, s=26, zorder=3,
                   label=PLANTED.label if i == 0 else None)
    ax.set_yticks(range(len(recs)), [f"{r['kind']}-on set,\n{LABEL[r['reading']]}" for r in recs], fontsize=7)
    ax.set_ylim(len(recs) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the split's energy")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc="lower right", frameon=False)


def sketch(out: Path = HERE / "figures") -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    data: list[dict[str, Any]] = []
    for ax, (title, rows) in zip(axes, SKETCH.items(), strict=True):
        recs = [{"kind": k, "reading": g, "planted": p, "random": r} for k, g, p, r in rows]
        draw(ax, recs, title)
        data += [{"panel": title, **r} for r in recs]
    axes[1].set_yticklabels([])  # the same rows as the left panel
    fig.suptitle("Planned figure (made-up numbers): does the split put a planted change where it was planted?", fontsize=9)
    FG.save(fig, out / "sketch_step1.png", n="made-up numbers; 648 planted subcomponents per kind over 28 down_proj matrices",
            data=data)


def regating(s: dict[str, Any], out: Path) -> None:
    """The re-gating share of each state against its matched size, the unscored ones left out."""
    fig, ax = FG.figure()
    styles = {"often": ("#0072B2", "o"), "rarely": ("#D55E00", "s"), "random": ("#555555", "X")}
    data = []
    for kind, (color, marker) in styles.items():
        pts = [(s["states"][f"{kind}_{z}"], z) for z in ("rl", "mid", "sft")]
        xy = [(p["kl"], p["regating"]) for p, _ in pts if p["status"] == "scored"]
        ax.plot([x for x, _ in xy], [y for _, y in xy], color=color, marker=marker, label=kind, linewidth=0.8)
        data += [{"kind": kind, "size": z, **p} for p, z in pts]
    ax.set_xscale("log")
    ax.set_xlabel("KL(W0 ‖ state), nats per token, 128 maths rows")
    ax.set_ylabel("re-gating share")
    ax.set_title(f"Re-gating share by size (E132's seed spread {s['seed_spread']:.4f})", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    FG.save(fig, out / "regating_step1.png", n="9 states, 128 maths rows of 256; unscored states not drawn", data=data)


def main(results: Path = HERE / "results" / "step1", out: Path = HERE / "figures") -> None:
    s = json.loads((results / "summary.json").read_text())
    for n, st in s["states"].items():
        st["kl"] = json.loads((results / f"S_{n}.json").read_text())["search"]["kl"]
    k = s["kinds"]
    recs = [{"kind": kind, "reading": g, "planted": k[kind][g], "random": k[kind]["chance" if g == "planted" else "on_null"]}
            for kind, g in ROWS]
    fig, ax = FG.figure((5.0, 3.6))
    draw(ax, recs, f"Split call: {s['calls']['split']['call']}")
    fig.suptitle("Where the least-squares split puts a planted change, beside a random one", fontsize=9)
    FG.save(fig, out / "split_step1.png", n="648 subcomponents per set over 28 down_proj matrices; one split per kind; new decomposition",
            data=recs)
    regating(s, out)


if __name__ == "__main__":
    sketch() if "--sketch" in sys.argv else main()
