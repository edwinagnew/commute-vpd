"""E136's one figure, from results/stage1.json only (no model): per pair, the share's gap between the side built to reuse and the side built to be new, on the trained
decomposition, each matrix's SVD and that SVD rotated, with the seeds behind each marker (each state's share less the other side's mean). Scored pairs first, then the
described ones, named as built; the pairs with no saved states are listed in the figure's note. No verdict of reused or new.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_untrained/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, Style, figure, save

HERE = Path(__file__).resolve().parent
# One colour and one marker per decomposition, as in the card's sketch (Okabe-Ito, as `commute.figures`).
DECOMPS = {"trained": Style("trained decomposition", "#D55E00", "D"), "svd": Style("each matrix's SVD", "#0072B2", "o"), "rotated": Style("SVD, randomly rotated", "#555555", "s")}
OFFSET = {"trained": 0.25, "svd": 0.0, "rotated": -0.25}


def seed_points(row: dict[str, Any], d: str) -> list[float]:
    """Each state's share, less the mean of the other side's: the gap this state would give with the other side averaged."""
    r, n = row[d]["reuse"], row[d]["new"]
    return [x - sum(n) / len(n) for x in r] + [sum(r) / len(r) - x for x in n]


def main() -> None:
    rec = json.loads((HERE / "results" / "stage1.json").read_text())
    rows = sorted(rec["pairs"], key=lambda r: not r["scored"])
    readable = [r for r in rows if r["readable"]]
    fig, ax = figure((FULL_WIDTH[0], 0.55 * len(readable) + 1.6))
    data = []
    for i, row in enumerate(reversed(readable)):
        for d, s in DECOMPS.items():
            y = i + OFFSET[d]
            pts = seed_points(row, d)
            ax.scatter(pts, [y] * len(pts), s=9, color=s.color, alpha=0.3, linewidths=0, zorder=1)
            ax.scatter([row[d]["gap"]], [y], s=42, color=s.color, marker=s.marker, edgecolors="white", linewidths=0.8, zorder=2, label=s.label if i == 0 else None)
            data.append({"pair": row["id"], "scored": row["scored"], "decomposition": d, "gap": row[d]["gap"], "shrink": row["shrink"]})
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(range(len(readable)), [f"{r['id']} ({r['n_reuse']} v {r['n_new']}{'' if r['scored'] else ', described'})" for r in reversed(readable)], fontsize=8)
    ax.set_xlabel("share's gap, side built to reuse minus side built to be new")
    v = rec["verdict"]
    ax.set_title(f"Filed reading: {v['verdict']} ({v['n_rival_a_side']} of {v['n_scored_readable']} scored pairs lose at least 0.1 on the SVD)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    not_read = "; ".join(x["id"] for x in rec["not_read"])
    unread = [r["id"] for r in rows if not r["readable"]]
    save(fig, HERE / "figures" / "gaps.png", n=f"not read, no states saved: {not_read}" + (f"; unreadable: {', '.join(unread)}" if unread else ""), data=data)


if __name__ == "__main__":
    main()
