"""The gate page's one figure (cell P1.1): per matched pair of old tasks, the found re-weighting share G of each seed on the
side built to reuse against the side built to be new, with the adopted G cut-off (0.8, user 2026-11-10) drawn. Reads results/labels.json and
labels_cut_0p8_0p5.json only (no model, no new computation).

Pairs: the three groups that share one input set (in-context copy against lookup, the real case label and the real
repeat label against their twins), and the two key-table pairs from one source each (copy against cipher on 24 keys,
identity against random on 32 keys), which are read on separate input sets drawn the same way.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/old_task_labels/plot_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.lines import Line2D

from commute import figures as FG

HERE = Path(__file__).resolve().parent
PAIRS = {  # row label -> (side built to reuse, side built to be new)
    "in-context copy | lookup": ("e62_copy", "e62_lookup"),
    "copy, 24 keys | cipher, 24 keys": ("copy24", "cipher24"),
    "identity, 32 keys | random, 32 keys": ("ident32", "rand32"),
    "real case label | twin": ("case_real", "case_twin"),
    "real repeat label | twin": ("repeat_real", "repeat_twin"),
}
SIDES = {"reuse": ("regating", -0.12), "new": ("neuron", 0.12)}  # METHODS styles borrowed for colour and marker only


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    items = json.loads((results / "labels.json").read_text())["items"]
    cut = json.loads((results / "labels_cut_0p8_0p5.json").read_text())["cutoffs_new"]["G"]  # adopted 2026-11-10 (decisions.md)
    g = {it["item"]: it["G"] for it in items}
    rows = []
    fig, ax = FG.figure((7.0, 3.4))
    for side, (method, offset) in SIDES.items():
        k = 0 if side == "reuse" else 1
        groups = {}
        for label, pair in PAIRS.items():
            task = pair[k]
            vals = [g[f"{task}_s{s}"] for s in (0, 1)]
            groups[label] = vals
            rows += [{"pair": label, "side": side, "task": task, "seed": s, "G": v} for s, v in enumerate(vals)]
        FG.dots(ax, groups, method=method, xlabel="re-weighting share G, best of 4 sites and 3 starts (0: nothing explained)", offset=offset)
    FG.reference(ax, cut, f"cut-off {cut:g}")
    ax.set_xlim(0, 1)
    handles = [Line2D([], [], color=FG.METHODS[m].color, marker=FG.METHODS[m].marker, linestyle="", label=f"side built {'to reuse' if s == 'reuse' else 'to be new'}")
               for s, (m, _) in SIDES.items()]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Every reuse side reads above its new twin; one of five clears the cut-off", fontsize=9, pad=14)
    FG.save(fig, out / "gate_within_pair.png", n="2 seeds per side, one dot per saved fine-tune; 8-token inputs", data=rows)


if __name__ == "__main__":
    main()
