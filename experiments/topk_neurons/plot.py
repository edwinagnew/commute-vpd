"""E217's figure from results/summary.json (no model): per threshold band, each of the 20 trained fine-tunes' overlap of
the strong-neuron sets of base and fine-tune, and its lowest site-remainder s (read positions), beside the reference
that the base read against itself (W0) and the full width both give, 1, and the filed range at p97/3.

  figures/overlap_remainder.png

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/topk_neurons/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as fg

HERE = Path(__file__).resolve().parent
BANDS = {"p99_1": "p99/1: top 2%", "p97_3": "p97/3: top 6%", "p90_10": "p90/10: top 20%"}
FILED = {"overlap": (0.7, 0.99), "s_rem": (0.99, 1.0)}  # predictions.json P5 and P4, filed for p97/3


def main() -> None:
    table = json.loads((HERE / "results" / "summary.json").read_text())["table"]
    preds = json.loads((HERE / "results" / "predictions.json").read_text())["predictions"]
    assert FILED == {"overlap": tuple(preds["P5"]["range"]), "s_rem": tuple(preds["P4"]["range"])}
    rows = [{"band": b, "item": k.split("/")[1], "overlap": r["overlap"], "s_rem_min_read": min(r["s_rem"][1])}
            for k, r in table.items() for b in BANDS if k.startswith(b + "/")]  # fmt: skip
    assert len(rows) == 60, len(rows)
    fig, axes = fg.figure((7.5, 3.4), ncols=2)
    for ax, key, xlabel in ((axes[0], "overlap", "overlap of base's and fine-tune's strong-neuron sets"),
                            (axes[1], "s_rem_min_read", "lowest site-remainder s, read positions")):  # fmt: skip
        lo, hi = FILED["overlap" if key == "overlap" else "s_rem"]
        ax.axvspan(lo, hi, color="#cccccc", alpha=0.5, label="filed range at p97/3")
        fg.dots(ax, {BANDS[b]: [r[key] for r in rows if r["band"] == b] for b in BANDS}, method="neuron", xlabel=xlabel)
        fg.reference(ax, 1.0, "base vs itself, full width")
        ax.legend(loc="center left", bbox_to_anchor=(0, 0.75), fontsize=7, frameon=False)
    axes[0].set_xlim(0, 1.05)
    axes[1].set_xlim(0.8, 1.01)
    axes[1].set_yticklabels([])
    fig.suptitle("Fine-tunes change which neurons fire strongly, and turn the weak remainder, more than filed")
    fg.save(fig, HERE / "figures" / "overlap_remainder.png", n="20 trained fine-tunes per band (10 tasks x 2 seeds), "
            "512 windows each; one dot per fine-tune", data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
