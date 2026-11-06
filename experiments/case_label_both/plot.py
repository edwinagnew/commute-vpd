"""E123's two figures.

route.png: each case-label fine-tune's least-squares split and its teacher-trained copy on E12's route axes (masked-off
share, re-switching share r), with E12's filed label lines, and as references E119's trigger fine-tunes and arm-A
copies and E12 Amendment 1's copy-table and random-map fine-tunes.
removal.png: per state, D(S) of the recruited set beside its 20 random same-size sets, with E38 Amendment 2's six seeds
as the reference and E38 A2's seed-rule lines (0.10 through, 0.05 not).

    python experiments/case_label_both/plot.py   -> figures/route.png, figures/removal.png
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
EXP = HERE.parent
E119 = EXP / "teacher_route_split" / "results"
E12A1 = EXP / "new_vs_existing" / "results" / "new_vs_existing_amend1.json"
E38A2 = EXP / "recruitment_test" / "results" / "amend2.json"
OUT = json.loads((RESULTS / "case_label_both.json").read_text())
LINES = json.loads((RESULTS / "predictions.json").read_text())["config"]["lines"]
STATES = ["real_s0", "real_s1", "twin_s0", "twin_s1"]
COLOUR = {"real": "#2a78d6", "twin": "#eb6834"}  # dataviz default slots 1-2, as E119's figure
INK, MUTED, REF, REGION = "#2b2b2b", "#8a8a85", "#b9b8ad", "#efeee9"


def point(shares: dict[str, float]) -> tuple[float, float]:
    return shares["e_gate"], shares["r"]


def regions(ax: Any) -> None:
    """E12's filed lines: re-switching r >= r_hi with e_gate <= masked_lo; masked-off e_gate >= masked_hi with r <= r_lo."""
    lo, hi = -0.3, 1.0
    ax.fill_between([lo, LINES["masked_lo"]], LINES["r_hi"], hi, color=REGION, zorder=0)
    ax.fill_between([LINES["masked_hi"], hi], lo, LINES["r_lo"], color=REGION, zorder=0)
    ax.text(lo + 0.02, hi - 0.03, "re-switching", fontsize=8, color=MUTED, va="top")
    ax.text(hi - 0.02, lo + 0.02, "masked-off", fontsize=8, color=MUTED, ha="right")
    ax.text(0.37, 0.95, "neither", fontsize=8, color=MUTED, ha="center")
    for x in (LINES["masked_lo"], LINES["masked_hi"]):
        ax.axvline(x, color=MUTED, lw=0.6, ls="--", zorder=1)
    for y in (LINES["r_lo"], LINES["r_hi"]):
        ax.axhline(y, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def references(ax: Any) -> None:
    for path in sorted(E119.glob("teacher_route_split_*.json")):
        out = json.loads(path.read_text())
        for opt, t in out["teachers"].items():
            ax.plot(*point(t["least_squares"]["shares"]), "D", ms=6, mfc="none", mec=REF, mew=1.2, zorder=2)
            s = out["students"].get(f"{opt}_A")
            if s is not None:
                ax.plot(*point(s["a"]["shares"]), "o", ms=6, color=REF, zorder=2)
    a1 = json.loads(E12A1.read_text())
    for c in a1["c0"].values():
        ax.plot(*point(c["least_squares"]["shares"]), "s", ms=6, mfc="none", mec="#6f6e66", mew=1.2, zorder=2)
    ax.annotate("E12 copy tables", (0.06, 0.81), xytext=(0.12, 0.88), fontsize=7, color=MUTED)
    ax.annotate("E12 random maps", (0.52, 0.27), xytext=(0.36, 0.36), fontsize=7, color=MUTED)
    ax.annotate("E119 trigger tasks", (0.0, 0.52), xytext=(-0.28, 0.40), fontsize=7, color=MUTED)


def route_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    regions(ax)
    references(ax)
    for n in STATES:
        colour = COLOUR[n.split("_")[0]]
        fx, fy = point(OUT["routes"][n]["least_squares"]["shares"])
        sx, sy = point(OUT["students"][n]["a"]["shares"])
        ax.plot([fx, sx], [fy, sy], color=colour, lw=1, alpha=0.6, zorder=3)
        ax.plot(fx, fy, "D", ms=8, mfc="white", mec=colour, mew=2, zorder=4)
        ax.plot(sx, sy, "o", ms=8, color=colour, mec="white", mew=1.5, zorder=5)
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=COLOUR["real"], label="real case-label arm (2 seeds)"),
               Line2D([], [], ls="", marker="o", ms=8, color=COLOUR["twin"], label="memorising twin (2 seeds)"),
               Line2D([], [], ls="", marker="D", ms=8, mfc="white", mec=INK, label="fine-tune, least-squares split"),
               Line2D([], [], ls="", marker="o", ms=8, color=INK, label="its teacher-trained copy"),
               Line2D([], [], ls="", marker="D", ms=6, mfc="none", mec=REF, label="E119 trigger fine-tunes"),
               Line2D([], [], ls="", marker="o", ms=6, color=REF, label="E119 copies (task windows)"),
               Line2D([], [], ls="", marker="s", ms=6, mfc="none", mec="#6f6e66", label="E12 A1 fine-tunes (copy tables, random maps)")]
    ax.legend(handles=handles, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False)
    ax.set_xlabel("masked-off share (gain carried by subcomponents Γ keeps off)")
    ax.set_ylabel("re-switching share r (gain from Γ switching released subcomponents on)")
    ax.set_title("Both case-label arms land in the masked-off corner, beside E12's random maps", fontsize=10)
    finish(fig, ax, "route.png")


def removal_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    a2 = json.loads(E38A2.read_text())["states"]
    names = [*STATES, "", *a2]
    for i, n in enumerate(names):
        if n in OUT["removal"]:
            rec, colour = OUT["removal"][n], COLOUR[n.split("_")[0]]
            rand, d_s, cls = rec["D_random"], rec["D_S"], rec["cls"]
        elif n in a2:
            rec, colour = a2[n]["D"]["trigger"], REF
            rand, d_s, cls = rec["random"], rec["S"], ""
        else:
            continue
        ax.plot([i] * len(rand), rand, "_", ms=12, mew=1.2, color=MUTED, zorder=2)
        ax.plot(i, d_s, "o", ms=10, color=colour, mec="white", mew=1.5, zorder=4)
        if cls:
            ax.annotate(cls, (i, d_s), xytext=(9, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0.10, color=MUTED, lw=0.7, ls="--")
    ax.axhline(0.05, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0.0, color=INK, lw=0.5)
    ax.text(4, 0.105, "through: ≥ 0.10\nand above all 20", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.text(4, 0.045, "not: below 0.05", fontsize=7, color=MUTED, ha="center", va="top")
    labels = ["real s0", "real s1", "twin s0", "twin s1", "", "emoticon s0", "emoticon s1", "r1 s0", "r1 s1", "r2 s0", "r2 s1"]
    ax.set_xticks(range(len(names)), labels, fontsize=8, rotation=30, ha="right")
    ax.text(1.5, -0.13, "E123: case-label pair", ha="center", fontsize=8, color=INK)
    ax.text(7.5, -0.13, "reference: E38 A2 trigger tasks", ha="center", fontsize=8, color=MUTED)
    ax.set_ylim(-0.16, 0.6)
    ax.set_xlim(-0.6, len(names) - 0.4)
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=COLOUR["real"], label="recruited set S, real arm"),
               Line2D([], [], ls="", marker="o", ms=9, color=COLOUR["twin"], label="recruited set S, twin"),
               Line2D([], [], ls="", marker="o", ms=9, color=REF, label="recruited set S, E38 A2"),
               Line2D([], [], ls="", marker="_", ms=12, mew=1.2, color=MUTED, label="20 random sets of the same size")]
    ax.legend(handles=handles, fontsize=7, loc="upper right", frameon=False)
    ax.set_ylabel("D: share of the gain lost by removing the set,\nfine-tune minus original model")
    ax.set_title("Does removing the switched-on set cost the fine-tune more than a random set?", fontsize=9)
    finish(fig, ax, "removal.png")


def finish(fig: Any, ax: Any, name: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150)
    print(FIGURES / name)


if __name__ == "__main__":
    route_figure()
    removal_figure()
