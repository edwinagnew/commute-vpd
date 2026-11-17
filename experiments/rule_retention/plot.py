"""E193 (cell P1.24) and amendment figures, from <results>/stage0.json, stage1.json and stage2.json (results/ for E193,
results/<amend>/ with `--amend <amend>`, figures then in figures/<amend>/):
    figures/profile.png   the retention profile: r of every unit of F not at r = 1, per item and level, with the count at 1
    figures/contrast.png  the retention for the contrast per item and level against the 0.3 and 0.7 bars; do-nothing, which
                          has no contrast, is drawn as the reference at r = 1

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention/plot.py [--amend amend1]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
LEVELS = {"A": "level A: MLP neurons", "B": "level B: heads and MLP blocks"}
COLOR = {"do-nothing": "#777777", "rescaled": "#0072B2", "replaced": "#D55E00"}


def items(s0: dict[str, Any], s1: dict[str, Any], s2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Item label -> readings by level, in the order drawn."""
    out = {"do-nothing": s0["readings"], "rescaled": s1["readings"]}
    return out | {f"replaced s{s}": v["readings"] for s, v in sorted(s2["seeds"].items(), key=lambda kv: int(kv[0]))}


def kind(label: str) -> str:
    return label.split(" ")[0]


def draw_profile(its: dict[str, dict[str, Any]], path: Path, note: str) -> None:
    fig, axes = figure((7.5, 2.0 + 1.1 * len(its)), ncols=2, nrows=len(its))
    data = []
    for i, (lab, rd) in enumerate(its.items()):
        for j, lv in enumerate("AB"):
            ax, v = axes[i][j], rd[lv]
            names, r = list(v["changed"]), list(v["changed"].values())
            ax.vlines(range(len(r)), 0, r, color=COLOR[kind(lab)], linewidth=1)
            ax.scatter(range(len(r)), r, color=COLOR[kind(lab)], s=12)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-1, max(len(r), 8))
            ax.set_xticks(range(len(r)), names, rotation=90, fontsize=5)
            ax.set_title(f"{lab}, {LEVELS[lv]}: {v['n_units'] - len(r)} of {v['n_units']} units at r = 1", fontsize=7)
            ax.set_ylabel("r", fontsize=7)
            data += [{"item": lab, "level": lv, "unit": n, "r": x} for n, x in zip(names, r, strict=True)]
    fig.suptitle(f"Retention r of each changed unit (dots); every other unit reads r = 1{note}")
    save(fig, path, n="one reading per unit on copy24's 75 held-out windows; replaced: one per fit seed", data=data)


def draw_contrast(panels: list[tuple[str, dict[str, dict[str, Any]]]], th: dict[str, Any], path: Path, note: str) -> None:
    """One panel per (title, items); per level a row of items, each the range over orders of the smallest sets' mean r."""
    fig, axes = figure((7.5, 3.4), ncols=len(panels))
    axes = axes if len(panels) > 1 else [axes]
    data = []
    for ax, (title, its) in zip(axes, panels, strict=True):
        labels = list(its)
        for j, lv in enumerate("AB"):
            for i, lab in enumerate(labels):
                y, c = i + 0.18 * (2 * j - 1), its[lab][lv]["contrast"]
                if c["reading"] == "no change":
                    ax.scatter([1.0], [y], facecolors="none", edgecolors=COLOR["do-nothing"], marker="o" if lv == "A" else "s", s=26)
                    continue
                if c["reading"] != "read":
                    ax.annotate(f"{lv}: {c['reading']}", (0.5, y), fontsize=6, ha="center", va="center")
                    continue
                lo, hi = min(c["retention"]), max(c["retention"])
                ax.plot([lo, hi], [y, y], color=COLOR[kind(lab)], linewidth=2)
                ax.scatter([sum(c["retention"]) / len(c["retention"])], [y], color=COLOR[kind(lab)], marker="o" if lv == "A" else "s", s=26)
                data += [{"panel": title, "item": lab, "level": lv, "order": k, "retention": x} for k, x in enumerate(c["retention"])]
        for x, t in ((th["low"], f"low bar {th['low']}"), (th["high"], f"high bar {th['high']}")):
            ax.axvline(x, color="#777777", linewidth=0.8, linestyle="--")
            ax.annotate(t, (x, 0.0), xycoords=("data", "axes fraction"), xytext=(2, 2), textcoords="offset points", fontsize=7, color="#555555")
        ax.axvspan(th["low"], th["high"], color="#eeeeee", zorder=0)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.05)
        ax.set_xlabel("retention for the contrast (mean r over a smallest set)", fontsize=8)
        ax.set_title(title, fontsize=8)
    fig.suptitle(f"Rescaled must read at or above 0.7, replaced at or below 0.3; grey band: no call{note}\n"
                 "circles: level A (MLP neurons); squares: level B (heads and MLP blocks)", fontsize=9)  # fmt: skip
    save(fig, path, n="one smallest set per random order (5); replaced: every fit seed; open markers: do-nothing, every r = 1", data=data)


def main(amend: str) -> None:
    res, figs = (RESULTS / amend, FIGURES / amend) if amend else (RESULTS, FIGURES)
    s = {n: json.loads((res / f"stage{n}.json").read_text()) for n in range(3)}
    pred = json.loads((res / "predictions.json").read_text())
    its = items(s[0], s[1], s[2])
    figs.mkdir(parents=True, exist_ok=True)
    draw_profile(its, figs / "profile.png", "")
    draw_contrast([(f"{pred['experiment']}, the three built items", its)], pred["thresholds"], figs / "contrast.png", "")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--amend") + 1] if "--amend" in sys.argv else "")
