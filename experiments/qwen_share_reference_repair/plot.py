"""The repair of P3.10's reference: its one figure, from results/summary.json (run.py --summary). Per row set, each
change's re-gating share minus the six-draw random curve at its own size K00 on p-015e0710 (MLP matrices only), with
the random band around 0: random draws, the library's fine-tunes and the 15 built changes (continuous size). Drawn by
E181's `gaps` (experiments/qwen_share_reference/plot.py), so the two figures read the same way. No reuse or new label.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_share_reference_repair/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from commute import figures as F
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
P = load_experiment("qwen_share_reference", "plot")  # E181's figure: gaps, CHOICE, FT, RND


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    # E181's labels read 'k=<count>'; here the size is the continuous x (whole neurons plus one turned down partway)
    s["built"] = [b | {"k": None if b["x"] is None else round(b["x"], 2)} for b in s["built"]]
    fig, ax = F.figure((F.FULL_WIDTH[0], 4.2))
    data: list[dict[str, Any]] = []
    for i, r in enumerate(s["rowsets"]):
        data += P.gaps(ax, i, r, s)
    ax.axhline(0, color=P.RND.color, lw=0.8)
    ax.set_xticks(range(len(s["rowsets"])), [f"{r} rows" for r in s["rowsets"]])
    ax.set_ylabel("share minus the random curve\nat the change's own size")
    ax.set_xlabel("row set")
    handles = [Line2D([], [], color=P.RND.color, marker=P.RND.marker, ls="", label="random change (faint: off size)"),
               Patch(color="#dddddd", label="random band: largest draw's distance from the curve")]
    handles += [Line2D([], [], color=c, marker=m, ls="", label=n) for n, c, m in P.FT.values()]
    handles += [Line2D([], [], color=c, marker=m, ls="", markeredgecolor="black", label=f"built: {n}")
                for n, c, m in P.CHOICE.values()]
    handles += [Line2D([], [], color="white", marker="o", ls="", markeredgecolor="#555555",
                       label="hollow: built change off its size target")]
    ax.legend(handles=handles, fontsize=6.5, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.set_title("Do switched-off neurons read further above the random curve than random changes do?", fontsize=9)
    F.save(fig, HERE / "figures" / "share_reference.png", data=data,
           n="Per row set: 6 random draws at each of 3 sizes, 3 built changes, the library's fine-tunes (6; 12 on "
             "gold rows), each read on 128 rows of 256 tokens, MLP matrices only. Built changes with no curve value "
             "at their size are not drawn")


if __name__ == "__main__":
    main()
