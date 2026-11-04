"""E76's figures from the results JSONs alone; no model.

  figures/two_terms.png    the clean term and the masked term against steps, each with its reference
  figures/size_vs_fall.png the masked term against the change's squared size, with δW's size marked
  figures/by_block.png     the change's squared size per block beside δW's
  figures/late_share.png   rule (iii) against steps, with the filed line and the control
  figures/parts.png        the fall in the masked term with only part of each change kept
  figures/hook.png         stage 0: the last-matrix shift with one cause of its gap swapped at a time

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_masked_teacher/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID, BAND = "#2b2a27", "#6f6e69", "#e4e3df", "#d9d8d3"
COLOUR = {"A": "#1baf7a", "B": "#2a78d6", "L": "#eb6834", "C": MUTED, "dW": "#2b2a27"}
NAME = {"A": "A: clean term alone", "B": "B: clean + masked, all 196 masked", "L": "L: clean + masked, library's routing",
        "C": "C: control, as B towards a 0.01-nat teacher"}
MARK = {"A": "s", "B": "o", "L": "D", "C": "^"}


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


ARMS = {a: load(f"{a}.json")["readings"] for a in "CABL"}
DW = load("dw_blocks.json")
OWN_MASKED = load("stage0.json")["K0_8_fp32"]


def style(ax: Axes, grid: Literal["x", "y"] = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str, bottom: float = 0.0) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def series(arm: str, key: str) -> tuple[list[int], list[float]]:
    rows = [r for r in ARMS[arm] if r[key] is not None]
    return [r["step"] for r in rows], [r[key] for r in rows]


def size(reading: dict) -> float:
    return sum(reading["blocks"]) / DW["sum"]


def two_terms() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=170, sharey=True, sharex=True)
    panels = (("i", "clean output: KL(FT clean ‖ student clean)"), ("iv", "masked output: KL(FT clean ‖ student masked), all 196 masked"))
    for ax, (key, title) in zip(axes, panels, strict=True):
        for arm in "CABL":
            xs, ys = series(arm, key)
            ax.plot(xs, ys, color=COLOUR[arm], lw=2, ls="--" if arm == "C" else "-", label=NAME[arm], marker=MARK[arm], ms=4, markevery=4)
        ax.set(title=title, xlabel="training step", yscale="log", ylim=(0.005, 3), xlim=(0, 1390), xticks=range(0, 1201, 200))
        ax.title.set_size(9.5)
        style(ax)
    left, right = axes
    left.axhline(0.09, color=INK, lw=1.2, ls=":")
    left.text(1385, 0.098, "filed bar 0.09", fontsize=8, color=INK, ha="right")
    left.text(1215, 0.012, "A 0.012", fontsize=8, color=INK, va="center")
    left.text(1215, 0.0245, "B 0.024\nL 0.023", fontsize=8, color=INK, va="center")
    left.text(30, 1.95, "W0 starts at 1.84", fontsize=8, color=INK)
    right.axhline(OWN_MASKED, color=INK, lw=1.2, ls=":")
    right.text(1385, OWN_MASKED * 0.74, "W0's own masked KL, W0 clean ‖ W0 masked: 0.042", fontsize=8, color=INK, ha="right")
    right.axvline(100, color=MUTED, lw=1, ls=":")
    right.text(112, 0.0065, "both masked arms already\nmeet the filed rule here", fontsize=8, color=MUTED)
    right.text(30, 1.5, "W0 starts at 1.33", fontsize=8, color=INK)
    for y, label in ((1.127, "A 1.13"), (0.39, "L 0.39"), (0.28, "B 0.28")):
        right.text(1215, y, label, fontsize=8, color=INK, va="center")
    left.set_ylabel("nats per position, log scale")
    handles, labels = left.get_legend_handles_labels()
    fig.suptitle("Does training on the masked term lower it, and how near does it get? Lower, not near", fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    finish(fig, "two_terms.png", 0.11)


def size_vs_fall() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=170)
    ax.axvline(1, color=INK, lw=1.2)
    ax.text(1.02, 1.75, "δW's squared size", fontsize=8, color=INK)
    ax.axhline(OWN_MASKED, color=INK, lw=1.2, ls=":")
    ax.axvline(1.5, color=MUTED, lw=1, ls=":")
    ax.text(1.52, 1.75, "1.5 × δW's, no rule reads this", fontsize=8, color=MUTED)
    ax.text(3.6, OWN_MASKED * 1.08, "W0's own masked KL 0.042", fontsize=8, color=INK, ha="right")
    for arm in "ABL":
        rows = ARMS[arm]
        xs, ys = [size(r) for r in rows], [r["iv"] for r in rows]
        ax.plot(xs, ys, color=COLOUR[arm], lw=2, label=NAME[arm], marker=MARK[arm], ms=3, zorder=3)
        for r in rows:
            if r["step"] in (500, 1200):
                ax.scatter(size(r), r["iv"], color=COLOUR[arm], marker=MARK[arm], s=55, edgecolor="white", zorder=4)
                if arm == "A":
                    ax.text(size(r) + 0.04, r["iv"], "A at step 1200" if r["step"] == 1200 else "", fontsize=7.5, color=INK, va="center")
                elif arm == "L":
                    ax.text(size(r), r["iv"] * 1.09, f"step {r['step']}", fontsize=7.5, color=INK, ha="center")
                else:
                    ax.text(size(r), r["iv"] * 0.87, f"step {r['step']}", fontsize=7.5, color=INK, ha="center")
    ax.set(xlabel="the change's squared size, over δW's", ylabel="KL(FT clean ‖ student masked), nats, log scale", yscale="log", ylim=(0.035, 2.1), xlim=(0, 3.65))
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center right", bbox_to_anchor=(1, 0.68))
    fig.suptitle("What does the masked fall cost in weight size? Well past δW's, and still growing at the last step", fontsize=11)
    finish(fig, "size_vs_fall.png")


def by_block() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.7), dpi=170)
    ax.axvspan(20.5, 27.5, color=BAND, alpha=0.5, zorder=0)
    ax.text(24, 0.0145, "blocks 21-27, read by the late-share rule", fontsize=8, color=INK, ha="center")
    x = list(range(28))
    ax.plot(x, DW["blocks"], color=COLOUR["dW"], lw=2, ls="--", marker="x", ms=5, label="δW, the update itself")
    for arm in "ABL":
        ax.plot(x, ARMS[arm][-1]["blocks"], color=COLOUR[arm], lw=2, marker=MARK[arm], ms=4, label=NAME[arm])
    last = ARMS["B"][-1]["blocks"][27]
    ax.text(27.2, last, "B 2.75,\n23.6 × δW's", fontsize=8, color=INK, va="center")
    ax.text(27.2, DW["blocks"][27] * 0.8, "δW 0.12", fontsize=8, color=INK, va="center")
    ax.set(xlabel="block", ylabel="squared size in the block, log scale", yscale="log", xlim=(-0.5, 30), ylim=(0.012, 4))
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Where does each change sit at step 1200? δW is flat; the masked arms rise towards the end", fontsize=11)
    finish(fig, "by_block.png")


def late_share() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.5), dpi=170)
    ax.axhline(3, color=INK, lw=1.2, ls=":")
    ax.text(1195, 3.05, "filed bar 3, no stated source", fontsize=8, color=INK, ha="right")
    ax.axhline(1, color=INK, lw=1.2)
    ax.text(1195, 1.05, "placed like δW = 1", fontsize=8, color=INK, ha="right")
    for arm in "CABL":
        xs, ys = series(arm, "iii")
        ax.plot(xs, ys, color=COLOUR[arm], lw=2, ls="--" if arm == "C" else "-", marker=MARK[arm], ms=4, markevery=2, label=NAME[arm])
    ax.annotate("C's teacher is a random change, yet C tracks B until it ends\nat step 300, where the plan expected it near A",
                xy=(300, ARMS["C"][-1]["iii"]), xytext=(370, 3.45), fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set(xlabel="training step", ylabel="late share of the change, over δW's", ylim=(0.8, 4.05), xlim=(0, 1210))
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 0.08))
    fig.suptitle("Does the late-share rule show the change is placed like the update? A random teacher reads the same as B", fontsize=11)
    finish(fig, "late_share.png")


def parts() -> None:
    a1, a2 = load("amend1.json"), load("amend2.json")
    base = a1["W0"]["iv"]
    rows = (("whole change", a1, "all"), ("blocks 0-14 alone", a2, "first"), ("blocks 15-20 alone", a2, "middle"),
            ("blocks 21-27 alone", a1, "late"), ("block 27 alone", a1, "last"))
    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=170)
    ax.axvline(base - OWN_MASKED, color=INK, lw=1.2, ls=":")
    ax.text(base - OWN_MASKED - 0.01, 2.6, "fall needed to reach\nW0's own masked KL, 1.29", fontsize=8, color=INK, ha="right", va="top")
    ax.axvline(0, color=INK, lw=1.2)
    offset = {"A": 0.2, "B": 0, "L": -0.2}
    labels = []
    for y, (name, src, key) in zip(range(4, -1, -1), rows, strict=True):
        for arm in "ABL":
            fall = base - src[arm][key]["iv"]
            ax.scatter(fall, y + offset[arm], color=COLOUR[arm], marker=MARK[arm], s=60, edgecolor="white", zorder=3, label=NAME[arm] if y == 4 else None)
            ax.text(fall + 0.018, y + offset[arm], f"{fall:.2f}", fontsize=7.5, color=INK, va="center")
        labels.append(f"{name}\nB's squared size {src['B'][key]['size']:.2f}")
    ax.set_yticks(range(4, -1, -1), labels, fontsize=8.5)
    ax.set(xlabel="fall in KL(FT clean ‖ student masked) from W0's 1.33, nats, with only this part of the change kept", xlim=(-0.03, 1.36), ylim=(-0.5, 4.5))
    style(ax, "x")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Which blocks lower the masked term? Mostly blocks 15-27, and the parts overlap, so they do not add up", fontsize=11)
    finish(fig, "parts.png")


def hook() -> None:
    h = load("stage0.json")["hook"]
    rows = (("Masked shift, rebuilt from saved vectors", "rebuild", INK, "o"),
            ("Denominator taken from the clean forward", "denominator_from_clean", COLOUR["B"], "s"),
            ("Part removed taken from the clean forward", "removed_from_clean", COLOUR["L"], "D"),
            ("Clean base for both, first order", "clean_first_order", MUTED, "^"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=170, sharey=True, sharex=True)
    for ax, key, name in zip(axes, ("amp", "cos"), ("logit-shift amplitude, masked over clean", "cosine of masked and clean logit shifts"), strict=True):
        ax.axvline(1, color=INK, lw=1.2)
        ax.text(0.985, 3.42, "clean = 1", fontsize=8, color=INK, ha="right")
        for y, (_, k, col, mark) in zip(range(3, -1, -1), rows, strict=True):
            ax.scatter(h[k][key], y, color=col, marker=mark, s=70, zorder=3, edgecolor="white")
            ax.text(h[k][key], y + 0.17, f"{h[k][key]:.2f}", fontsize=8, color=INK, ha="center")
        ax.set(xlabel=name, xlim=(0.4, 1.08), ylim=(-0.5, 3.7))
        style(ax, "x")
    axes[0].set_yticks(range(3, -1, -1), [r[0] for r in rows], fontsize=8.5)
    fig.suptitle("Why did the last-matrix control read below 1? The final norm: its denominator sets the size, the part removed the angle", fontsize=10.5)
    finish(fig, "hook.png")


if __name__ == "__main__":
    two_terms()
    size_vs_fall()
    by_block()
    late_share()
    parts()
    hook()
