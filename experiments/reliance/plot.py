"""E283's control figure (no GPU), from results/summary.json: on the edits' cue windows, V1 (share of F's flip mass on
variables M also needs) and V2 (share on needed variables whose write barely moved) for the two strengthening edits,
the two new edits and the matched-KL random, with each item's construction floor for V1 (1 minus the flip share on
the variables the edit changed: what V1 reads at least if M needs every unchanged variable F needs).

    ~/commute-agents/fleet/cpu-run python3 experiments/reliance/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import ONE_COLUMN, figure, save

HERE = Path(__file__).resolve().parent
ITEMS = [("control_cue", "x2", "weights x2"), ("control_cue", "x10", "weights x10"), ("control", "new_few", "new, few"),
         ("control", "new_thin", "new, thin"), ("control", "random_kl_241002", "random,\nmatched KL")]


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    by = {(r["group"], r["item"]): r for r in s["items"] if r["feat"] == "neurons"}
    rows = [{"item": name, **{k: by[(g, i)][k] for k in ("V1", "V2", "v1_floor")}} for g, i, name in ITEMS]
    fig, ax = figure(ONE_COLUMN)
    x = range(len(rows))
    ax.scatter([i - 0.12 for i in x], [r["V1"] for r in rows], color="#c0392b", label="V1: needed by both", zorder=3)
    ax.scatter([i + 0.12 for i in x], [r["V2"] for r in rows], color="#e1a100", marker="D", label="V2: needed, barely changed", zorder=3)
    for i, r in zip(x, rows):
        ax.hlines(r["v1_floor"], i - 0.3, i + 0.3, color="#333333", linewidth=1.2, label="V1 construction floor" if i == 0 else None)
    ax.set_xticks(list(x), [r["item"] for r in rows], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of F's flip mass")
    ax.set_title("Do the strengthening edits read above the new ones?", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    save(fig, HERE / "figures" / "v1_control.png", n="one built edit each, cue windows (512, last position), neurons as variables", data=rows)


if __name__ == "__main__":
    main()
