"""The case-label line's closing figure, from saved results only, no model.

  figures/summary_causal.png  per causal form run on the case-label pair (E112 stage 2, E112 Amendments 1 and 2, E113),
                              the per-state reading against the filed bars (real at least 0.50, twin at most 0.20,
                              random control at most 0.10), with the largest random-control reading in grey; beside
                              it, separated, E62's copying head (2,4) off on the copy and lookup states, which is a
                              located-head removal and not a form the definition names.

Removal and erasure read the share of the label gain removed; the patch reads the share of labels flipped. Both are
read against the same filed bars. Colours as E112's and E113's plot.py: real blue, twin orange, controls grey, seed 1
hatched; copy green, lookup purple.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/case_patch/plot_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EXP = Path(__file__).parent.parent
CL, CP, IC = EXP / "case_label" / "results", EXP / "case_patch" / "results", EXP / "induction_cue" / "results"
FIG = EXP / "case_patch" / "figures"
BLUE, ORANGE, GREEN, PURPLE, GREY, GRID = "#2a78d6", "#eb6834", "#1f9d6b", "#8b5cc8", "#7a7a75", "#e4e3df"
CASE = [("real_s0", BLUE, ""), ("real_s1", BLUE, "//"), ("twin_s0", ORANGE, ""), ("twin_s1", ORANGE, "//")]
COPY = [("copy_s0", GREEN, ""), ("copy_s1", GREEN, "//"), ("copy_s2", GREEN, ".."),
        ("lookup_s0", PURPLE, ""), ("lookup_s1", PURPLE, "//"), ("lookup_s2", PURPLE, "..")]  # fmt: skip
W = 0.15


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def case_groups() -> list[tuple[str, dict[str, float] | None, float]]:
    """(label, per-state reading or None if not read, largest random-control reading over the four states)."""
    s2, a1, a2, pt = load(CL / "stage2.json"), load(CL / "amend1.json"), load(CL / "amend2.json"), load(CP / "run.json")
    worst = lambda d: max(v["share"] for v in d.values())
    return [
        ("remove one direction\nblock 1, trigger\n(E112)", {s: s2["states"][s]["P3"] for s, *_ in CASE}, worst(s2["do_nothing"])),
        ("remove two directions\nblocks 1-3, every position\n(E112 Amendment 1)", {s: a1["states"][s]["P3"] for s, *_ in CASE}, worst(a1["do_nothing"])),
        ("erase case, 148 directions\nblocks 1-3\n(E112 Amendment 2)", None, worst(a2["do_nothing"])),
        ("patch to the other\ncase's mean, block 1\n(E113)", {s: pt["states"][s]["Q1"] for s, *_ in CASE}, max(max(v) for v in pt["control"].values())),
    ]  # fmt: skip


def draw_case(ax: Any, x0: float, vals: dict[str, float] | None, rnd: float, bar_max: float) -> None:
    for i, (s, c, h) in enumerate(CASE):
        x = x0 + (i - 2) * W
        if vals is not None:
            ax.bar(x, vals[s], W, color=c, hatch=h, edgecolor="white")
            ax.text(x, vals[s] + 0.01, f"{vals[s]:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    xr = x0 + 2 * W
    ax.bar(xr, rnd, W, color=GREY, edgecolor="white")
    ax.text(xr, rnd + 0.01, f"{rnd:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.hlines(bar_max, xr - W / 2 - 0.02, xr + W / 2 + 0.02, color=GREY, ls="--", lw=1.2)
    if vals is None:
        ax.text(x0 - 0.5 * W, 0.42, "not read:\nits random\ncontrol was\nover 0.10", ha="center", va="center", fontsize=7, color=GREY)


def draw_copy(ax: Any, x0: float) -> None:
    st = load(IC / "run.json")["states"]
    for i, (s, c, h) in enumerate(COPY):
        x = x0 + (i - 3) * W
        v = st[s]["share"]
        ax.bar(x, v, W, color=c, hatch=h, edgecolor="white")
        ax.text(x, max(v, 0) + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    rnd = max(max(st[s]["random_shares"]) for s, *_ in COPY)
    xr = x0 + 3 * W
    ax.bar(xr, rnd, W, color=GREY, edgecolor="white")
    ax.text(xr, rnd + 0.01, f"{rnd:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rules = load(CL / "predictions.json")["rules"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    groups = case_groups()
    for k, (_, vals, rnd) in enumerate(groups):
        draw_case(ax, k, vals, rnd, rules["do_nothing_max"])
    xc = len(groups) + 0.35
    draw_copy(ax, xc)
    ax.axvline(len(groups) - 0.42, color=GRID, lw=1.5)
    ax.axhline(rules["real"]["P3"], color=BLUE, ls="--", lw=1.2)
    ax.axhline(rules["twin_expected_max"]["P3"], color=ORANGE, ls=":", lw=1.2)
    labels = [g[0] for g in groups] + ["beside, not a form the\ndefinition names: copying\nhead (2,4) off (E62)"]
    ax.set(xticks=list(range(len(groups))) + [xc], xticklabels=labels, xlim=(-0.6, xc + 0.75), ylim=(-0.03, 1.0),
           ylabel="share removed: label gain (removal, head off)\nor labels flipped (patch)",
           title="Does any causal check take the case-label fine-tune's behaviour away?")  # fmt: skip
    ax.tick_params(axis="x", labelsize=7.5)
    handles = [Patch(facecolor=BLUE, label="case label, real (seed 1 hatched)"),
               Patch(facecolor=ORANGE, label="case-label twin (seed 1 hatched)"),
               Patch(facecolor=GREEN, label="copy at a marker, seeds 0-2"),
               Patch(facecolor=PURPLE, label="lookup at the marker, seeds 0-2"),
               Patch(facecolor=GREY, label="random control, largest reading"),
               Line2D([], [], color=BLUE, ls="--", label="filed bar for 'reuses' (0.50)"),
               Line2D([], [], color=ORANGE, ls=":", label="twin expected at most (0.20)"),
               Line2D([], [], color=GREY, ls="--", label="random control allowed at most (0.10)")]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "summary_causal.png", dpi=150)


if __name__ == "__main__":
    main()
