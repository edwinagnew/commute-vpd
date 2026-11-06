"""E119's one figure: each student's (masked-off share, re-switching share r) beside its fine-tune's least-squares split,
on E12's route axes, with E12's filed label regions and E12 Amendment 1's states in grey as the reference.

    python experiments/teacher_route_split/plot.py   -> figures/route_labels.png (reads results/teacher_route_split_*.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
E12A1 = HERE.parent / "new_vs_existing" / "results" / "new_vs_existing_amend1.json"
CM = json.loads((RESULTS / "predictions.json").read_text())["config"]
COLOUR = {"r1_t534_v369": "#2a78d6", "r0_t352_v310": "#eb6834", "emoticon": "#1baf7a"}  # dataviz default slots 1-3
MARKER = {"A": "o", "B": "s"}
INK, MUTED, REGION = "#2b2b2b", "#8a8a85", "#efeee9"


def point(rs: dict[str, Any]) -> tuple[float, float]:
    return rs["shares"]["e_gate"], rs["shares"]["r"]


def regions(ax: Any) -> None:
    """E12's filed lines: re-switching r >= r_hi with e_gate <= masked_lo; masked-off e_gate >= masked_hi with r <= r_lo."""
    lo, hi = -0.3, 1.0  # the r0 and r1 SGD fine-tunes read masked-off shares down to -0.24
    ax.fill_between([lo, CM["masked_lo"]], CM["r_hi"], hi, color=REGION, zorder=0)
    ax.fill_between([CM["masked_hi"], hi], lo, CM["r_lo"], color=REGION, zorder=0)
    ax.text(lo + 0.02, hi - 0.03, "re-switching", fontsize=8, color=MUTED, va="top")
    ax.text(hi - 0.02, lo + 0.02, "masked-off", fontsize=8, color=MUTED, ha="right")
    ax.text(0.5, 0.95, "neither", fontsize=8, color=MUTED, ha="center")
    for x in (CM["masked_lo"], CM["masked_hi"]):
        ax.axvline(x, color=MUTED, lw=0.6, ls="--", zorder=1)
    for y in (CM["r_lo"], CM["r_hi"]):
        ax.axhline(y, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def e12_reference(ax: Any) -> None:
    a1 = json.loads(E12A1.read_text())
    for n, s in a1["students"].items():
        if "_o" not in n:
            ax.plot(*point(s["a"]), "o", ms=5, color="#c3c2b7", zorder=2)
    for c in a1["c0"].values():
        ax.plot(*point(c["least_squares"]), "D", ms=5, mfc="none", mec="#c3c2b7", zorder=2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    regions(ax)
    e12_reference(ax)
    for task, colour in COLOUR.items():
        path = RESULTS / f"teacher_route_split_{task}.json"
        if not path.is_file():
            continue
        out = json.loads(path.read_text())
        for opt, t in out["teachers"].items():
            if "least_squares" not in t:
                continue
            fx, fy = point(t["least_squares"])
            ax.plot(fx, fy, "D", ms=8, mfc="white", mec=colour, mew=2, zorder=4)
            ax.annotate(opt.upper() if opt == "sgd" else "Adam", (fx, fy), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
            for arm in ("A", "B"):
                s = out["students"].get(f"{opt}_{arm}")
                if s is None:
                    continue
                sx, sy = point(s["a"])
                ax.plot([fx, sx], [fy, sy], color=colour, lw=1, alpha=0.6, zorder=3)
                ax.plot(sx, sy, MARKER[arm], ms=8, color=colour, mec="white", mew=1.5, zorder=5)
                if not out.get("checks", {}).get(f"{opt}_{arm} student P(target) against E34's (difference)", {}).get("pass", True):
                    ax.plot(sx, sy, "o", ms=17, mfc="none", mec="#d03030", mew=1.5, zorder=6)
                    ax.annotate(f"failed reproduction:\nP(target) {s['p_target']:.2f}, E34 read 0.90;\nthe job stopped here;\ngain {s['gain_ratio']:.2f} of the teacher's: unreadable",
                                (sx, sy), xytext=(-20, -62), textcoords="offset points", fontsize=7, color="#d03030")
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=c, label=t) for t, c in COLOUR.items()]
    handles += [Line2D([], [], ls="", marker="o", ms=8, color=INK, label="student, arm A (task windows)"),
                Line2D([], [], ls="", marker="s", ms=8, color=INK, label="student, arm B (Pile rows)"),
                Line2D([], [], ls="", marker="D", ms=8, mfc="white", mec=INK, label="its fine-tune (least-squares split)"),
                Line2D([], [], ls="", marker="o", ms=5, color="#c3c2b7", label="E12 A1 students (copy top left, maps bottom right)"),
                Line2D([], [], ls="", marker="D", ms=5, mfc="none", mec="#c3c2b7", label="E12 A1 fine-tunes"),
                Line2D([], [], ls="", marker="o", ms=12, mfc="none", mec="#d03030", label="student off E34 by more than 0.05")]
    ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=False)
    ax.set_xlabel("masked-off share (gain carried by subcomponents Γ keeps off)")
    ax.set_ylabel("re-switching share r (gain from Γ switching released subcomponents on)")
    ax.set_title("Where each teacher-trained decomposition carries its gain at the trigger", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "route_labels.png", dpi=150)
    print(FIGURES / "route_labels.png")


if __name__ == "__main__":
    main()
