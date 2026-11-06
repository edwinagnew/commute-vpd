"""E112's figures from results/stage{0,1,2}.json, amend1.json and predictions.json, no model.
Real task blue, twin orange, the original model and the random-direction control grey; seed 1 hatched.

  figures/possession.png  per block, the case probe's and the Hewitt-Liang control probe's accuracy on held-out
                          task windows at the trigger, with the 0.90 bar and the 0.30 selectivity bar
  figures/carrying.png    per state, label accuracy on held-out words and agreement with a threshold on the
                          original model's case readout, with the real bar and the twin's expected ceiling
  figures/removal.png     per state, the share of the label gain removed by taking the case direction out at block 1,
                          beside the same share for a random direction, with the 0.50 bar
  figures/beside.png      held-out label accuracy after pruning the top-K MLP units against K random ones, and during
                          reverse fine-tuning, for the real fine-tunes (no rule)
  figures/amend1_removal.png  Amendment 1 (results/amend1.json): per state, the share removed by stage 2's removal,
                          by the previous-word direction alone and by both case directions at every position of blocks
                          1-3, and by two random directions there, with the 0.50, 0.20 and 0.10 bars

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/case_label/plot.py
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

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"
STATES = [("real_s0", "real, seed 0", BLUE, ""), ("real_s1", "real, seed 1", BLUE, "//"),
          ("twin_s0", "twin, seed 0", ORANGE, ""), ("twin_s1", "twin, seed 1", ORANGE, "//")]  # fmt: skip


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def possession(s0: dict[str, Any], rules: dict[str, Any]) -> None:
    P0, blocks = s0["P0"], range(len(s0["P0"]["acc_case"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    w = 0.38
    ax.bar([b - w / 2 for b in blocks], P0["acc_case"], w, color=BLUE, label="case probe (trained on Pile text)")
    ax.bar([b + w / 2 for b in blocks], P0["acc_control"], w, color=GREY, label="control probe (random bit per word pair)")
    ax.axhline(rules["P0"]["accuracy"], color=INK, ls="--", lw=1)
    ax.text(3.45, rules["P0"]["accuracy"] + 0.01, "accuracy bar 0.90", ha="right", va="bottom", color=INK, fontsize=8)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(3.45, 0.51, "chance", ha="right", va="bottom", color=GREY, fontsize=8)
    for b, (a, sel) in enumerate(zip(P0["acc_case"], P0["selectivity"], strict=True)):
        ax.text(b - w / 2, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(b - w / 2, 0.06, f"gap\n{sel:.2f}", ha="center", fontsize=8, color="white")
    ax.annotate("first passing block", (s0["l_star"] - w / 2, 1.04), ha="center", fontsize=8, color=BLUE)
    ax.set(xticks=list(blocks), xticklabels=[f"block {b}" for b in blocks], ylim=(0, 1.3),
           ylabel="accuracy on held-out task words at the trigger",
           title="Does the original model hold 'previous word is capitalised'?\n(gap = case minus control; bar 0.30)")  # fmt: skip
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=1)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "possession.png", dpi=150)


def state_bars(ax: Any, x0: float, vals: list[float], w: float) -> None:
    for i, ((_, _, c, h), v) in enumerate(zip(STATES, vals, strict=True)):
        ax.bar(x0 + (i - 1.5) * w, v, w, color=c, hatch=h, edgecolor="white")
        ax.text(x0 + (i - 1.5) * w, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)


def bar_lines(ax: Any, x0: float, real: float, twin: float, half: float) -> None:
    ax.hlines(real, x0 - half, x0 + half, color=BLUE, ls="--", lw=1.2)
    ax.hlines(twin, x0 - half, x0 + half, color=ORANGE, ls=":", lw=1.4)


def carrying(s0: dict[str, Any], s2: dict[str, Any], rules: dict[str, Any]) -> None:
    S, w = s2["states"], 0.18
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for x0, key in ((0, "P1"), (1, "P2")):
        state_bars(ax, x0, [S[s][key] for s, *_ in STATES], w)
        bar_lines(ax, x0, rules["real"][key], rules["twin_expected_max"][key], 2.4 * w)
    ax.bar(-0.75, s0["w0_heldout_accuracy"]["real"], w, color=GREY)
    ax.text(-0.75, s0["w0_heldout_accuracy"]["real"] + 0.01, "original\nmodel", ha="center", va="bottom", fontsize=7)
    ax.set(xticks=[0, 1], xlim=(-1.0, 1.55), ylim=(0, 1.12), ylabel="share of held-out-word windows",
           xticklabels=["label right on held-out words", "label agrees with a threshold\non the original model's case readout"],
           title="Does the fine-tune carry to unseen words, as a threshold on the case readout?")  # fmt: skip
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=n) for _, n, c, h in STATES]
    handles += [Line2D([], [], color=BLUE, ls="--", label="bar for 'reuses' (0.90)"),
                Line2D([], [], color=ORANGE, ls=":", label="twin expected at most (0.65)")]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "carrying.png", dpi=150)


def removal(s2: dict[str, Any], rules: dict[str, Any]) -> None:
    S, D, w = s2["states"], s2["do_nothing"], 0.18
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    state_bars(ax, 0, [S[s]["P3"] for s, *_ in STATES], w)
    for i, (s, _, c, h) in enumerate(STATES):
        ax.bar(1 + (i - 1.5) * w, D[s]["share"], w, color=GREY, hatch=h, edgecolor="white")
        ax.text(1 + (i - 1.5) * w, D[s]["share"] + 0.01, f"{D[s]['share']:.4f}", ha="center", va="bottom", fontsize=6, rotation=90)
    bar_lines(ax, 0, rules["real"]["P3"], rules["twin_expected_max"]["P3"], 2.4 * w)
    ax.hlines(rules["do_nothing_max"], 1 - 2.4 * w, 1 + 2.4 * w, color=GREY, ls="--", lw=1.2)
    ax.set(xticks=[0, 1], xlim=(-0.6, 1.6), ylim=(0, 0.6), ylabel="share of the label gain removed",
           xticklabels=["case direction removed\n(block 1 output, trigger position)", "random direction removed\n(same place)"],
           title="Does removing the case direction take away the fine-tune's label gain?")  # fmt: skip
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=n) for _, n, c, h in STATES]
    handles += [Line2D([], [], color=BLUE, ls="--", label="bar for 'reuses' (0.50)"),
                Line2D([], [], color=ORANGE, ls=":", label="twin expected at most (0.20)"),
                Line2D([], [], color=GREY, ls="--", label="random direction allowed at most (0.10)")]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper right", ncol=1)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "removal.png", dpi=150)


def amend1(a1: dict[str, Any], rules: dict[str, Any]) -> None:
    S, D, w = a1["states"], a1["do_nothing"], 0.18
    groups = [[S[s]["P3_stage2"] for s, *_ in STATES], [S[s]["P3_prev_alone"] for s, *_ in STATES],
              [S[s]["P3"] for s, *_ in STATES]]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for x0, vals in enumerate(groups):
        state_bars(ax, x0, vals, w)
        bar_lines(ax, x0, rules["real"]["P3"], rules["twin_expected_max"]["P3"], 2.4 * w)
    for i, (s, _, _, h) in enumerate(STATES):
        v = D[s]["share"]
        ax.bar(3 + (i - 1.5) * w, v, w, color=GREY, hatch=h, edgecolor="white")
        ax.text(3 + (i - 1.5) * w, max(v, 0) + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.hlines(rules["do_nothing_max"], 3 - 2.4 * w, 3 + 2.4 * w, color=GREY, ls="--", lw=1.2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(xticks=[0, 1, 2, 3], xlim=(-0.6, 3.6), ylim=(-0.05, 0.6), ylabel="share of the label gain removed",
           xticklabels=["previous-word case\ndirection, block 1,\ntrigger only (stage 2)",
                        "same direction,\nblocks 1-3,\nevery position",
                        "previous-word and\nown-word case directions,\nblocks 1-3, every position",
                        "two random directions,\nblocks 1-3,\nevery position"],
           title="Does a stronger removal of the case readout take away the label gain?")  # fmt: skip
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=n) for _, n, c, h in STATES]
    handles += [Line2D([], [], color=BLUE, ls="--", label="bar for 'reuses' (0.50)"),
                Line2D([], [], color=ORANGE, ls=":", label="twin expected at most (0.20)"),
                Line2D([], [], color=GREY, ls="--", label="random directions allowed at most (0.10)")]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_removal.png", dpi=150)


def beside(s2: dict[str, Any]) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for s, n, c, h in STATES[:2]:
        P, ks = s2["pruning"][s], [1, 4, 16, 64]
        ls = "-" if not h else "--"
        a.plot(ks, [P[str(k)]["top"] for k in ks], color=c, ls=ls, marker="o", label=f"{n}: top-K units")
        a.plot(ks, [min(P[str(k)]["random"]) for k in ks], color=GREY, ls=ls, marker="s", label=f"{n}: K random, lowest of 5")
        cur = s2["reverse"][s]["curve"]
        b.plot([t for t, _ in cur], [v for _, v in cur], color=c, ls=ls, marker="o", label=n)
    for ax in (a, b):
        ax.axhline(0.5, color=GREY, ls=":", lw=1)
        style(ax)
    b.axhline(0.6, color=INK, ls="--", lw=1)
    b.text(60, 0.585, "reverse fine-tune stop (0.6)", ha="right", va="top", fontsize=8, color=INK)
    for ax, x in ((a, 64), (b, 60)):
        ax.text(x, 0.505, "chance", ha="right", va="bottom", fontsize=8, color=GREY)
    a.set(xscale="log", xticks=[1, 4, 16, 64], xticklabels=["1", "4", "16", "64"], xlabel="MLP units zeroed (K)",
          ylabel="label accuracy on held-out words", ylim=(0.45, 1.03), title="Pruning the most important MLP units")  # fmt: skip
    b.set(xlabel="reverse fine-tuning steps", title="Fine-tuning back to the original next tokens")
    a.legend(fontsize=7, frameon=False, loc="lower left")
    b.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Beside the verdict, real fine-tunes only: how easily is the label behaviour taken away?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "beside.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rules = load("predictions.json")["rules"]
    s0, s2 = load("stage0.json"), load("stage2.json")
    possession(s0, rules)
    carrying(s0, s2, rules)
    removal(s2, rules)
    beside(s2)
    amend1(load("amend1.json"), rules)


if __name__ == "__main__":
    main()
