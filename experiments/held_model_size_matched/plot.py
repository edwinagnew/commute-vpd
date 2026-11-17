"""P1.27 figures, from results/summary.json (no GPU, no model):
    figures/gaps.png    two panels, reading 3 (the per-site map) and reading 1 (one gain per unit), never pooled: per pair,
                        the reuse side's mean S minus the new side's, at the saved sizes and with the larger side matched
                        in size (E187's c); n = seeds per side. 0 = no difference.
    figures/shifts.png  the same two readings: per pair, each larger-side state's S at the matched size minus its S at
                        its own size, one point per state; the vertical line marks 0, "size made no difference".

    ~/commute-agents/fleet/cpu-run python experiments/held_model_size_matched/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, dots, figure, gap_heatmap, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
READINGS = {"3": "reading 3, per-site map", "1": "reading 1, one gain per unit"}
N_TEXT = "2 to 8 seeds per side (E187's saved states); 512 read windows per item, last position; fits on 16,384 fit windows"


def gaps(summary: dict[str, Any]) -> None:
    fig, axs = figure(FULL_WIDTH, ncols=2)
    rows = []
    for ax, (r, title) in zip(axs, READINGS.items(), strict=True):
        vals, counts = [], []
        for pair, p in summary["pairs"].items():
            g = p[f"reading {r}"]
            vals.append([None, None] if g.get("missing") else [g["gap_original"], g["gap_matched"]])
            n = None if g.get("missing") else len(g["S_original"]["reuse"]) + len(g["S_original"]["new"])
            counts.append([n, n])
            rows += [{"reading": r, "pair": pair, "size": k, "gap": v} for k, v in zip(("saved", "matched"), vals[-1], strict=True)]
        gap_heatmap(ax, vals, list(summary["pairs"]), ["saved sizes", "sizes matched"], label="S, reuse side minus new side",
                    fmt=".4f", counts=counts)  # fmt: skip
        ax.set_title(title, fontsize=8)
    save(fig, FIGURES / "gaps.png", n=N_TEXT, data=rows)


def shifts(summary: dict[str, Any]) -> None:
    fig, axs = figure(FULL_WIDTH, ncols=2)
    rows = []
    for ax, (r, title) in zip(axs, READINGS.items(), strict=True):
        groups = {f"{pair} (c {p['c']:.2f})": [] if p[f"reading {r}"].get("missing") else p[f"reading {r}"]["shift_by_state"]
                  for pair, p in summary["pairs"].items()}  # fmt: skip
        dots(ax, groups, method="size", xlabel="S at matched size minus S at saved size, same state")
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_title(title, fontsize=8)
        rows += [{"reading": r, "pair": k, "shift": v} for k, vs in groups.items() for v in vs]
    save(fig, FIGURES / "shifts.png", n=N_TEXT, data=rows)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    gaps(summary)
    shifts(summary)


if __name__ == "__main__":
    main()
