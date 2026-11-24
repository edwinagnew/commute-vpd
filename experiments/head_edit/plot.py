"""E280's figure (no GPU), from results/summary.json (run.py --stage summary): E_b over the needed set K per built edit
of head (2, 4), at VPD and at the neuron/head featurization side by side, with the card's bands (rival A at 0.5, rival
B under 0.3, the neuron positive control at 0.9). An edit not read yet is left out.

    ~/commute-agents/fleet/cpu-run python3 experiments/head_edit/plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commute.figures import METHODS, figure, save

HERE = Path(__file__).resolve().parent
EDITS = {"v2": "V x2", "o2": "O x2", "q2": "Q x2"}
FEATS = {"vpd": ("VPD subcomponents", "regating", -0.12), "neurons": ("neurons and heads", "neuron", 0.12)}
LINES = {0.9: ("neuron control bar (0.9)", ":"), 0.5: ("rival A bar (0.5)", "--"), 0.3: ("rival B bar (0.3)", "-.")}


def main(summary: Path, out: Path) -> None:
    table = json.loads(summary.read_text())["table"]
    fig, ax = figure()
    data = []
    for f, (label, method, dx) in FEATS.items():
        st = METHODS[method]
        pts = [(i + dx, table[e][f]) for i, e in enumerate(EDITS) if table.get(e, {}).get(f) is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=st.color, marker=st.marker, s=30, label=label, linewidths=0)
        data += [{"edit": e, "feat": f, "E_b_K": table.get(e, {}).get(f)} for e in EDITS]
    for y, (label, ls) in LINES.items():
        ax.axhline(y, color="#777777", linewidth=0.8, linestyle=ls, label=label)
    ax.set_xticks(range(len(EDITS)), list(EDITS.values()))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(EDITS) - 0.5)
    ax.set_ylabel("E_b over the needed set K")
    ax.set_xlabel("built edit of head (2, 4), everything else as M")
    ax.legend(fontsize=6, frameon=False, loc="center right")
    ax.set_title("One-head edits read at VPD and at neurons/heads", fontsize=8)
    save(fig, out, n="one built state per edit, one reading per featurization; E62's copy D, 512 windows, read at the marker", data=data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=HERE / "results" / "summary.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "eb_head_edits.png")
    a = ap.parse_args()
    main(a.summary, a.out)
