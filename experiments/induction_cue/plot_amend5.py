"""E62 Amendment 5's figures, from results/amend5.json and predictions_amend5.json (no model).

  figures/amend5_heads.png  each of the 24 heads: target model's fall against masked forward's fall, whole range on equal axes and the same x zoomed in y,
                            with the do-nothing reference, the line where the mask keeps the target's fall, the filed band and the all-masks-at-1 control
  figures/amend5_swap.png   masked forward's log-odds of B under the four mixes of tokens and masks and with head (2,4) off, with the same three levels for every mask at 1 and for the target model

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/induction_cue/plot_amend5.py
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
FIG, RES = HERE / "figures", HERE / "results"
BLOCK = {"0": ("#2a78d6", "o"), "1": ("#eb6834", "s"), "2": ("#1baf7a", "D"), "3": ("#eda100", "^")}  # the dataviz palette's fixed order; a shape too, so block is not colour alone
TARGET, MASKED = "#1baf7a", "#eda100"  # as amend3's and amend4's figures
ONES = "#2a78d6"
INK, GRID, GREY = "#52514e", "#e4e3df", "#8a8983"
ZOOM_NAMED = {"1,1": (0.0, 0.07, "center"), "2,4": (0.0, 0.07, "center"), "2,3": (0.12, 0.04, "left"), "3,4": (0.14, -0.02, "left"), "1,4": (0.0, -0.11, "center"), "0,4": (0.0, -0.12, "center"), "1,2": (0.0, 0.05, "center"), "3,5": (0.08, -0.12, "center")}


def style(ax: Any) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def points(ax: Any, heads: dict[str, Any], bars: bool) -> None:
    for name, r in heads.items():
        colour, marker = BLOCK[name[0]]
        if bars:
            ax.errorbar(r["T"], r["M"], xerr=r["T_sd"], yerr=r["M_sd"], color=colour, lw=1, zorder=3)
        ax.scatter(r["T"], r["M"], s=62, color=colour, marker=marker, edgecolors="white", linewidths=1.2, zorder=4)


def heads_figure(out: dict[str, Any], th: dict[str, Any]) -> None:
    heads, control, f = out["stage_c_sweep"]["heads"], out["stage_a_control"], out["verdict"]["f_labelled"]
    fig, (whole, zoom) = plt.subplots(1, 2, figsize=(13.6, 6.0), width_ratios=[1, 1.15])
    lo, hi = -3.2, 5.6
    for ax in (whole, zoom):
        style(ax)
        ax.axhline(0, color=GREY, lw=1, zorder=1)
        ax.axvline(0, color=GREY, lw=1, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("target model's fall with the head off, log-odds of B")
    whole.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=1.2, zorder=2)
    whole.text(1.9, 2.6, "mask keeps the\ntarget's fall", color=INK, fontsize=9, ha="right")
    points(whole, heads, False)
    whole.scatter(heads["2,4"]["T"], control["fall"], s=120, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=5)
    whole.text(heads["2,4"]["T"] - 0.35, control["fall"] - 0.1, "(2,4) with every\nmask at 1", color=INK, fontsize=9, ha="right")
    whole.text(heads["2,4"]["T"] - 0.15, heads["2,4"]["M"] + 0.3, "(2,4)", color=INK, fontsize=9, ha="right")
    whole.text(heads["1,1"]["T"] - 0.1, heads["1,1"]["M"] - 0.55, "(1,1)", color=INK, fontsize=9, ha="right")
    whole.annotate("do-nothing:\nno head off, 0 and 0", (0, 0), (-3.0, 1.6), color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8})
    whole.set_xlim(lo, hi)
    whole.set_ylim(lo, hi)
    whole.set_aspect("equal")
    whole.set_ylabel("masked forward's fall, log-odds of B")
    whole.set_title("Whole range, equal axes", color=INK, fontsize=10.5)
    gap, floor = th["fraction_gap"], th["fall_floor"]
    xs = [floor, hi]
    zoom.fill_between(xs, [(f - gap) * x for x in xs], [(f + gap) * x for x in xs], color=GRID, alpha=0.7, zorder=1)
    zoom.text(2.35, 0.08, "filed band: (2,4)'s fraction,\nplus or minus 0.24", color=INK, fontsize=9, ha="left")
    zoom.plot(xs, [f * x for x in xs], color=INK, ls=":", lw=1.2, zorder=2)
    zoom.axvline(floor, color=GREY, ls=":", lw=1, zorder=1)
    zoom.text(floor + 0.05, 1.07, "heads right of this\nline are compared", color=INK, fontsize=8.5, va="top")
    zoom.text(-3.1, 0.03, "do-nothing level", color=INK, fontsize=8.5, va="bottom")
    points(zoom, heads, True)
    for name, (dx, dy, ha) in ZOOM_NAMED.items():
        zoom.text(heads[name]["T"] + dx, heads[name]["M"] + dy, f"({name})", color=INK, fontsize=9, ha=ha)
    zoom.set_ylim(-0.32, 1.15)
    zoom.set_title("Same x, y zoomed in; bars are window bootstrap sds", color=INK, fontsize=10.5)
    handles = [Line2D([], [], color="none", marker=m, markerfacecolor=c, markeredgecolor="white", markersize=8, label=f"block {b}") for b, (c, m) in BLOCK.items()]
    zoom.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.8), frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle("With each of 24 heads off in turn, how much of the target model's fall does W₀'s masked forward keep?", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "amend5_heads.png", dpi=150)
    plt.close(fig)


def swap_figure(out: dict[str, Any]) -> None:
    a, d, c = out["stage_a_reread"], out["stage_d_swapped"], out["stage_a_control"]
    groups = [("masked forward, W₀'s masks", MASKED, [("intact tokens, intact windows' mask", a["mask_on"], False), ("pair-broken tokens, intact windows' mask", d["intact_gates_on_broken"], False),
                                                     ("(2,4) off, intact tokens, intact windows' mask", a["mask_off"], True), ("intact tokens, pair-broken windows' mask", d["broken_gates_on_intact"], False),
                                                     ("pair-broken tokens, pair-broken windows' mask", a["mask_broken"], False)]),
              ("same forward, every mask at 1", ONES, [("intact tokens", c["on"], False), ("(2,4) off, intact tokens", c["off"], True), ("pair-broken tokens", c["broken"], False)]),
              ("target model, no mask", TARGET, [("intact tokens", a["target_on"], False), ("(2,4) off, intact tokens", a["target_off"], True), ("pair-broken tokens", a["target_broken"], False)])]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    style(ax)
    ax.grid(axis="y", visible=False)
    i, labels = 0, []
    for name, colour, rows in groups:
        ax.text(-13.1, i - 0.46, name, color=INK, fontsize=8.5, va="top", weight="bold")
        if i:
            ax.axhline(i - 0.5, color=GREY, lw=1)
        for label, v, head_off in rows:
            ax.scatter(v, i, s=120, color="white" if head_off else colour, edgecolors=colour if head_off else "white", linewidths=1.8 if head_off else 1.5, zorder=4)
            ax.text(v + 0.18, i, f"{v:.2f}", color=INK, va="center", fontsize=9.5)
            labels.append(label)
            i += 1
    ax.set_yticks(range(len(labels)), labels, color=INK, fontsize=9.5)
    ax.set_ylim(i - 0.5, -0.7)
    ax.set_xlim(-13.2, -3.4)
    ax.set_xlabel("mean log-odds of B at the second A, 200 windows")
    ax.set_title("In the masked forward, which moves B's log-odds: the tokens or the mask?", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "amend5_swap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out = json.loads((RES / "amend5.json").read_text())
    th = json.loads((RES / "predictions_amend5.json").read_text())["thresholds"]
    heads_figure(out, th)
    swap_figure(out)


if __name__ == "__main__":
    main()
