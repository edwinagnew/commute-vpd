"""E214's figure: R under the main grade (read positions) per item, reuse by construction against new by construction,
one panel per level, from results/substitute.json. No GPU:

    python experiments/kstar_levels/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as fg

HERE = Path(__file__).resolve().parent
GROUPS = (("reuse: E195 gains x2, x10, moved, rebuilt", ("x2", "x10", "moved", "rebuilt"), "regating"),
          ("new: built edits, few and thin", ("new_few", "new_thin"), "neuron"),
          ("new: five grafted heads", ("e201", "last_lowest", "last_rank5", "last_rank9", "last_rank13"), "neuron"),
          ("new: 256 grafted neurons", ("neurons",), "neuron"))  # fmt: skip
LEVELS = {"head_mlp": "heads and MLP layers", "mlp_attn": "MLP and attention layers", "block": "blocks"}


def main() -> None:
    sub = json.loads((HERE / "results" / "substitute.json").read_text())
    g = sub["main_grade"]
    fig, axes = fg.figure(size=(7.5, 6.2), nrows=3)
    rows = []
    for ax, (L, title) in zip(axes, LEVELS.items(), strict=True):
        v = sub[f"grade_{g}"][L]
        for i, (name, items, method) in enumerate(GROUPS):
            fg.dots(ax, {name: [v["R"][it] for it in items]}, method=method, xlabel="R, mean grade over the minimal circuit", offset=i)
            rows += [{"level": L, "group": name, "item": it, f"R_{g}": v["R"][it]} for it in items]
        ax.set_yticks(range(len(GROUPS)), [n for n, _, _ in GROUPS], fontsize=7)
        lo = v["min_reuse"] - v["band_new_edits"]
        ax.axvspan(lo, v["min_reuse"], color="#cccccc", alpha=0.5, label="lowest reuse item minus band")
        fg.reference(ax, 1.0, "do-nothing, R = 1")
        ax.set_title(f"{title}: reuse minus built new {v['gap_edits']:+.3f}, band {v['band_new_edits']:.3f}", fontsize=8)
        ax.set_xlim(0.82, 1.005)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Reuse items read above the built new edits at every level, by less than the band at two", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fg.save(fig, HERE / "figures" / "r_by_construction.png", n="one item per point; built edits on 512 windows read at the "
            "last position, grafts on 512 windows and E195 items on 75 windows read at every position; grade at the read positions", data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
