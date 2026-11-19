"""Run 3 (E212) figure: the crosscoder's reading on its two controls, against the stop bar and E203's built-pair values.

    ~/commute-agents/fleet/cpu-run python experiments/realistic_tasks/plot3.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures

HERE = Path(__file__).parent
E203 = HERE.parent / "crosscoder" / "results" / "run2" / "stage1_summary.json"


def main() -> None:
    gauss = json.loads((HERE / "results" / "run3_gauss.json").read_text())["side"]["reads"]
    planted = json.loads((HERE / "results" / "run3_planted.json").read_text())["side"]["reads"]["L1"]
    built = json.loads(E203.read_text())
    stop = json.loads((HERE / "results" / "predictions_run3.json").read_text())["thresholds"]["full"]["gauss_stop"]
    rows = {
        "Gaussian update, French rows\n(the stop rule's rows)": ("random", [gauss["T2"]["d_authors"]]),
        "Gaussian update, the other\nfive sides' rows": ("random", [gauss[t]["d_authors"] for t in ("L1", "T1", "T3", "T4")]),
        "planted change, label rows": ("neuron", [planted["d_authors"]]),
        "E203: random update,\nbuilt pair": ("random", [built["random"]["d_authors"]]),
        "E203: trained sides,\nbuilt pair": ("neuron", [built[k]["d_authors"] for k in ("reuse", "new", "reuse_seed1")]),
    }
    fig, ax = figures.figure(figures.FULL_WIDTH)
    for method in ("random", "neuron"):
        figures.dots(ax, {k: v if m == method else [] for k, (m, v) in rows.items()}, method=method,
                     xlabel="crosscoder reading: shared part minus fine-tune-specific part of the activation change")
    figures.reference(ax, stop, f"stop bar {stop}")
    ax.axvline(0, color="#999999", linewidth=0.6)
    ax.set_title("Does a random update read below trained updates on the crosscoder?")
    data = [{"row": k.replace("\n", " "), "d_authors": x} for k, (_, v) in rows.items() for x in v]
    figures.save(fig, HERE / "figures" / "run3_controls.png", data=data,
                 n="one crosscoder per row group; the label rows are shared by the topic and coin sides, so the Gaussian's"
                   " other-rows group has four distinct values; E203 values from its stage 1 (first built pair)")


if __name__ == "__main__":
    main()
