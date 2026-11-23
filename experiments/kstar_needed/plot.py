"""E268 (cell P1.54) figure from results/summary.json (run.py --stage summary): figures/eb_k.png.

Left: E_b over K (the needed set, single-variable interchange) against E_b over K* (the saved mean-ablation circuit),
one dot per saved state, coloured by group, with y = x and the card's seed spread 0.04 around it. The one state whose
K is empty is drawn apart: its E_b over K is 1 by convention. Right: |K| / |K*| per group, every state a dot (the empty
K left out: log axis). Writes .png, .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/kstar_needed/plot.py [SUMMARY] [OUT_DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from commute import figures as FG

HERE = Path(__file__).resolve().parent
SPREAD = 0.04  # card.md: seed sd of the E220 rl arm's E_b over K*
# group name -> the figures.METHODS style that gives its colour and marker (the style's own label is not used)
GROUPS = {
    "built edits": "spectrum",
    "repeat task, RL": "neuron",
    "repeat task, SFT": "regating",
    "repeat task, shuffled-reward RL": "head",
    "procedural tasks, RL": "steering",
    "procedural tasks, guarded RL": "prior",
    "procedural tasks, SFT": "neuron_half",
    "matched random updates": "random",
}
BUILT_NAMES = {
    "x2": ("gain x2 and x10", (-6, 4), "right"),
    "rebuilt": ("rebuilt", (6, -10), "left"),
}


def group(item: str) -> str:
    """rl/stage2b/repeat_own_s1 -> "repeat task, SFT"; p256/addition_randg_s0 -> "matched random updates"."""
    if item.startswith("rl/"):
        arm = item.split("repeat_")[1].rsplit("_s", 1)[0]
        return {
            "rl": "repeat task, RL",
            "demo": "repeat task, SFT",
            "own": "repeat task, SFT",
            "shuffled": "repeat task, shuffled-reward RL",
        }[arm]
    if item.startswith("p256/"):
        arm = item.rsplit("_", 2)[-2]
        return {
            "rl": "procedural tasks, RL",
            "rlg": "procedural tasks, guarded RL",
            "sft": "procedural tasks, SFT",
        }.get(arm, "matched random updates")
    return "built edits"


def scatter(ax: Any, rows: list[dict[str, Any]]) -> None:
    grid = np.geomspace(0.005, 1.3, 200)
    ax.fill_between(
        grid,
        np.maximum(grid - SPREAD, 1e-4),
        grid + SPREAD,
        color="#bbbbbb",
        alpha=0.35,
        linewidth=0,
        label=f"within {SPREAD} of equal (seed spread)",
    )
    ax.plot(
        grid, grid, color="#555555", linewidth=0.8, linestyle="--", label="equal: y = x"
    )
    for name, method in GROUPS.items():
        st = FG.METHODS[method]
        pts = [r for r in rows if r["group"] == name and not r["K_empty"]]
        ax.scatter(
            [r["E_b_Kstar"] for r in pts],
            [r["E_b_K"] for r in pts],
            color=st.color,
            marker=st.marker,
            s=24,
            alpha=0.85,
            linewidths=0,
            label=f"{name} ({len(pts)})",
        )
    for r in rows:
        if r["K_empty"]:
            ax.scatter([r["E_b_Kstar"]], [r["E_b_K"]], facecolors="none", edgecolors="#D55E00", marker="o", s=60, linewidths=1.2,
                       label="K empty (collapsed RL state): 1 by convention")  # fmt: skip
        if r["item"] in BUILT_NAMES:
            text, off, ha = BUILT_NAMES[r["item"]]
            ax.annotate(
                text,
                (r["E_b_Kstar"], r["E_b_K"]),
                xytext=off,
                textcoords="offset points",
                ha=ha,
                fontsize=7,
            )
    ax.set(xscale="log", yscale="log", xlim=(0.006, 1.3), ylim=(0.006, 1.3))
    ax.set_xlabel("write share over K* (mean-ablation circuit, saved)")
    ax.set_ylabel("write share over K (needed set, this run)")
    ax.set_title(
        "Over K the write share repeats its value over K*,\nexcept on repeat-task RL",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="upper left", frameon=False)


def sizes(ax: Any, rows: list[dict[str, Any]]) -> None:
    shown = [r for r in rows if not r["K_empty"]]
    for name, method in GROUPS.items():
        FG.dots(ax, {g: [r["K_over_Kstar"] for r in shown if r["group"] == g and g == name] for g in GROUPS}, method=method,
                xlabel="|K| / |K*|, variables needed over variables in K*")  # fmt: skip
    ax.set_xscale("log")
    FG.reference(ax, 1.0, "|K| = |K*|")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(
        "K is about as large as K* on built edits,\nmostly smaller on trained states",
        fontsize=9,
    )


def main(summary: Path, out_dir: Path) -> None:
    items = json.loads(summary.read_text())["items"]
    rows = [{"item": r["item"], "group": group(r["item"]), "E_b_K": r["E_b_K"], "E_b_Kstar": r["E_b_Kstar"], "size_K": r["size_K"],
             "size_Kstar": r["size_Kstar"], "size_both": r["size_both"], "K_over_Kstar": r["size_K"] / r["size_Kstar"],
             "K_empty": r["size_K"] == 0, "n_read_positions": r["n_read_positions"], "min_margin": r["min_margin"]}
            for r in items.values() if r.get("reading") == "read"]  # fmt: skip
    fig, (a, b) = FG.figure((10.0, 4.8), ncols=2)
    scatter(a, rows)
    sizes(b, rows)
    n_empty = sum(r["K_empty"] for r in rows)
    FG.save(fig, out_dir / "eb_k.png", data=rows,
            n=f"{len(rows)} saved states, one dot each; K from 256 pairs of inputs per state (4 derangements of 64). "
              f"Right panel leaves out the {n_empty} state with K empty (log axis).")  # fmt: skip


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json",
        Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures",
    )
