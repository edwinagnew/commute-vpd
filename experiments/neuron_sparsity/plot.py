"""E63's figures from results/{weights,gradients,windows,stage0}.json and E56's job1, job2, no model. One dot per seed.
Built to elicit is green, built to be new is orange, as E54 and E56; the cipher at its own step is the darker orange.

  figures/update_sparsity.png   the weight update's spread over neurons, and the first-batch gradient's, per task, each
                                against its reference (rotated value; same batch with shuffled targets) and the control
  figures/rescale_share.png     share of the update along each neuron's own W0 weights, against the Gaussian's and the
                                rescaling control's
  figures/activation_sparsity.png  spread over neurons of one window's activation shift, against a Gaussian update's and
                                the rotated value, with the 16-neuron control
  figures/c1_family_a.png       share of the gain against k, per-window ranks beside E56's mean ranks and a random order
  figures/c1_families_bc.png    the same share curves on families B and C, two panels with shared axes

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_sparsity/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
E56 = HERE.parent / "keydependent_patch" / "results"
GREEN, ORANGE, DARK, GREY = "#1f9e6e", "#eb6834", "#9c3d14", "#7a7a75"
INK, GRID = "#52514e", "#e4e3df"
K = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288]
K56 = [1, 4, 16, 64, 256, 1024, 4096, 12288]
TICKS = [1, 4, 16, 64, 256, 1024, 4096, 12288]

# Row label, state prefix, seeds, colour. Top to bottom in the figures; "beside" rows are in no test.
TASKS = [
    ("A: elicit map, step 10", "elicit", 5, GREEN),
    ("A: cipher, same update size, step 9", "cipher", 5, ORANGE),
    ("A: cipher, own learned step 30", "cipher_own", 2, DARK),
    ("B: same word back, step 10", "ident32", 2, GREEN),
    ("B: random word, step 10", "rand32", 2, ORANGE),
    ("B beside: same word, other keys, step 10", "ident32b", 2, GREEN),
    ("B beside: random word, other keys, step 30", "rand32b", 2, DARK),
    ("C: lowercase, step 300", "lowercase", 2, GREEN),
    ("C: 500-word cipher, step 300", "broad_cipher", 2, ORANGE),
    ("C beside: 50-word cipher, step 300", "e53_cipher", 2, ORANGE),
]


def load(name: str) -> dict[str, dict[str, Any]]:
    return json.loads((RES / f"{name}.json").read_text())["rows"]


def states(prefix: str, seeds: int) -> list[str]:
    return [f"{prefix}_s{i}" for i in range(seeds)]


def setup(ax: Axes) -> None:
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def rows(ax: Axes, labels: bool) -> None:
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([t[0] for t in TASKS], fontsize=9, color=INK)
    ax.tick_params(axis="y", labelleft=labels)
    ax.set_ylim(len(TASKS) - 0.5, -1.0)
    for y in (2.5, 6.5):
        ax.axhline(y, color=GRID, lw=1.2)


def dots(ax: Axes, data: dict[str, dict[str, Any]], key: str, marker: str, filled: bool, dy: float = 0.0,
         skip_own: bool = False) -> None:
    """One mark per seed in its task's row; filled for the reading, open for its reference."""
    for y, (_, prefix, seeds, colour) in enumerate(TASKS):
        if skip_own and prefix == "cipher_own":
            continue
        xs = [data[s][key] for s in states(prefix, seeds)]
        offsets = np.linspace(-0.12, 0.12, seeds) + dy
        ax.scatter(xs, y + offsets, s=38, marker=marker, zorder=3, linewidths=1.3, edgecolors=colour,
                   facecolors=colour if filled else "white")


def vline(ax: Axes, x: float, text: str, ha: str = "left", colour: str = INK, y: float = -0.95) -> None:
    ax.axvline(x, color=colour, ls="--", lw=1.1, zorder=1)
    ax.text(x, y, " " + text + " ", fontsize=8, color=INK, ha=ha, va="top", zorder=4,
            bbox={"fc": "white", "ec": "none", "pad": 0.5})


def legend(fig: Any, handles: list[Line2D], ncol: int = 3) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=9, labelcolor=INK)


def mark(label: str, marker: str, filled: bool, colour: str = INK) -> Line2D:
    return Line2D([], [], ls="", marker=marker, markersize=7, markeredgecolor=colour, label=label,
                  markerfacecolor=colour if filled else "white")


KIND_MARKS = [mark("built to elicit", "o", True, GREEN), mark("built to be new", "o", True, ORANGE),
              mark("the new task trained further", "o", True, DARK)]


def update_sparsity(w: dict[str, Any], g: dict[str, Any], s0: dict[str, Any]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 6.6), sharey=True, width_ratios=[1, 1])
    for ax in (left, right):
        setup(ax)
    rows(left, True)
    rows(right, False)
    dots(left, w, "W1", "o", True, -0.1)
    dots(left, w, "W1_rot", "o", False, 0.14)
    left.set_xlim(0.78, 1.005)
    left.set_xlabel("spread of the finished update over the 12288 neurons (1 = even)", color=INK)
    left.set_title("Finished update: flat for every task", color=INK, fontsize=11)
    vline(left, s0["synthetic"]["w1_gaussian"], "unit Gaussian", ha="right")
    vline(left, w["code_step10"]["W1"], "update that learned nothing", ha="right", colour=GREY, y=-0.75)
    left.text(0.782, 9.4, "the 16-neuron control reads 0.0013, far off this axis", fontsize=8, color=INK)

    dots(right, g, "W1g", "s", True, -0.1, skip_own=True)
    dots(right, g, "W1g_shuf", "s", False, 0.14, skip_own=True)
    right.set_xscale("log")
    right.set_xlim(0.001, 1.1)
    right.set_xlabel("spread of the first-batch gradient over neurons, log axis", color=INK)
    right.set_title("First gradient: narrow for every task, shuffled targets too", color=INK, fontsize=11)
    vline(right, s0["synthetic"]["w1_sixteen"], "16-neuron control")
    vline(right, s0["synthetic"]["w1_gaussian"], "unit Gaussian", ha="right")
    right.text(0.0015, 2.08, "same gradient as the matched cipher's seeds 0, 1", fontsize=8, color=INK, va="center")
    fig.suptitle("Is the update itself on fewer neurons when built to elicit? No: elicit is on the wider side",
                 color=INK, fontsize=13)
    legend(fig, [*KIND_MARKS, mark("finished update", "o", True), mark("its rotated value", "o", False),
                 mark("first-batch gradient", "s", True), mark("same batch, targets shuffled", "s", False)], ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "update_sparsity.png", dpi=150)
    plt.close(fig)


def rescale_share(w: dict[str, Any], s0: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    setup(ax)
    rows(ax, True)
    dots(ax, w, "W2", "o", True, -0.1)
    dots(ax, w, "W2_gauss", "o", False, 0.14)
    ax.set_xscale("log")
    ax.set_xlim(0.0007, 1.3)
    ax.set_xlabel("share of the update's energy along each neuron's own W0 weights, log axis", color=INK)
    vline(ax, s0["synthetic"]["w2_sixteen"], "control: 16 neurons rescaled reads 1", ha="right")
    vline(ax, 1 / 768, "chance, 1/768", ha="right")
    vline(ax, w["code_step10"]["W2"], "update that learned nothing", colour=GREY, y=-0.75)
    ax.set_title("Do fine-tunes rescale the neurons W0 already has? No task does: all sit next to chance", color=INK,
                 fontsize=12)
    legend(fig, [*KIND_MARKS, mark("the fine-tune", "o", True), mark("Gaussian update of the same norms", "o", False)],
           ncol=5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "rescale_share.png", dpi=150)
    plt.close(fig)


def activation_sparsity(a: dict[str, Any], s0: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    setup(ax)
    rows(ax, True)
    dots(ax, a, "A1", "o", True, 0.0)
    dots(ax, a, "A1_gauss", "^", False, -0.25)
    dots(ax, a, "A1_rot", "D", False, 0.25)
    ax.set_xscale("log")
    ax.set_xlim(0.0004, 0.4)
    ax.set_xlabel("spread over the 12288 neurons of one window's activation shift, median window, log axis", color=INK)
    vline(ax, s0["sixteen"]["A1"], "16-neuron control")
    ax.set_title("Is the activation shift on fewer neurons when built to elicit? A and C yes, B no", color=INK,
                 fontsize=12)
    legend(fig, [*KIND_MARKS, mark("the fine-tune", "o", True), mark("Gaussian update of the same norms", "^", False),
                 mark("the fine-tune's shift, rotated", "D", False)], ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "activation_sparsity.png", dpi=150)
    plt.close(fig)


def curve(data: dict[str, Any], names: list[str], key: str, grid: list[int]) -> np.ndarray:
    return np.array([[data[s][key][str(k)] for k in grid] for s in names])


def draw(ax: Axes, data: dict[str, Any], names: list[str], key: str, grid: list[int], colour: str, ls: str, lw: float,
         band: bool = True) -> None:
    arr = curve(data, names, key, grid)
    ax.plot(grid, arr.mean(0), color=colour, ls=ls, lw=lw, zorder=3)
    if band:
        ax.fill_between(grid, arr.min(0), arr.max(0), color=colour, alpha=0.18, lw=0, zorder=2)


def crossing(ax: Axes, a: dict[str, Any], names: list[str], colour: str) -> None:
    """The filed count: the mean log2 k where the curve first reaches half of the state's own full share."""
    c1, half = np.mean([a[s]["C1"] for s in names]), np.mean([a[s]["full_share"] for s in names]) / 2
    ax.scatter([2**c1], [half], s=70, color=colour, edgecolors="white", linewidths=1.2, zorder=5)


def axes_k(ax: Axes) -> None:
    setup(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TICKS, [str(k) for k in TICKS])
    ax.minorticks_off()
    ax.set_xlim(1, 12288)
    ax.set_ylim(-0.07, 1.03)
    ax.set_xlabel("neurons patched per window, k", color=INK)
    ax.axhline(0, color=INK, lw=0.8)


def c1_family_a(a: dict[str, Any], s0: dict[str, Any]) -> None:
    e56 = {**json.loads((E56 / "job1.json").read_text())["states"],
           **json.loads((E56 / "job2.json").read_text())["states"]}
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    axes_k(ax)
    for prefix, seeds, colour in (("elicit", 5, GREEN), ("cipher", 5, ORANGE), ("cipher_own", 2, DARK)):
        names = states(prefix, seeds)
        draw(ax, a, names, "shares", K, colour, "-", 2.2)
        draw(ax, e56, names, "b2_shares", K56, colour, "--", 1.5, band=False)
        if prefix != "cipher_own":
            draw(ax, a, names, "shares_random", K, colour, ":", 1.5, band=False)
        crossing(ax, a, names, colour)
    sixteen = s0["sixteen"]["shares"]
    ax.plot(K, [sixteen[str(k)] for k in K], color=GREY, lw=1.6, zorder=3)
    ax.axhline(0.5, color=INK, ls="--", lw=0.9)
    ax.text(1.1, 0.51, "half of the whole gain, E56's target", fontsize=8, color=INK, va="bottom")
    ax.text(1.1, -0.06, "0 = original model", fontsize=8, color=INK, va="bottom")
    ax.set_ylabel("share of the fine-tune's gain on the scoring windows (1 = fine-tuned)", color=INK)
    ax.set_title("How many neurons carry the gain? Ranking per window cuts the count; the elicit-cipher gap "
                 "closes with progress", color=INK, fontsize=11.5)
    handles = [Line2D([], [], color=GREEN, lw=2.2, label="elicit map, step 10"),
               Line2D([], [], color=ORANGE, lw=2.2, label="cipher, same update size, step 9"),
               Line2D([], [], color=DARK, lw=2.2, label="cipher, own learned step 30"),
               Line2D([], [], color=GREY, lw=1.6, label="control: update on 16 neurons"),
               Line2D([], [], color=INK, lw=2.2, label="ranked per window, this run (band: seed range)"),
               Line2D([], [], color=INK, ls="--", lw=1.5, label="ranked by mean shift, the earlier run"),
               Line2D([], [], color=INK, ls=":", lw=1.5, label="random order, this run"),
               mark("half of the state's own full share: the filed count", "o", True)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "c1_family_a.png", dpi=150)
    plt.close(fig)


def c1_families_bc(a: dict[str, Any]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    for ax in (left, right):
        axes_k(ax)
    panels = ((left, "Family B: elicit is the wider side", (("ident32", GREEN, "-"), ("rand32", ORANGE, "-"),
                                                            ("ident32b", GREEN, "--"), ("rand32b", DARK, "--"))),
              (right, "Family C: elicit is much the narrower side", (("lowercase", GREEN, "-"),
                                                                     ("broad_cipher", ORANGE, "-"),
                                                                     ("e53_cipher", ORANGE, "--"))))
    for ax, title, arms in panels:
        for prefix, colour, ls in arms:
            names = states(prefix, 2)
            draw(ax, a, names, "shares", K, colour, ls, 2.2 if ls == "-" else 1.4, band=ls == "-")
            crossing(ax, a, names, colour)
            if ls == "-":
                draw(ax, a, names, "shares_random", K, colour, ":", 1.4, band=False)
        ax.set_title(title, color=INK, fontsize=11)
    left.set_ylabel("share of the fine-tune's gain on the scoring windows (1 = fine-tuned)", color=INK)
    fig.suptitle("Does the same count hold on other pairs? The direction flips between them", color=INK, fontsize=13)
    handles = [Line2D([], [], color=GREEN, lw=2.2, label="built to elicit"),
               Line2D([], [], color=ORANGE, lw=2.2, label="built to be new"),
               Line2D([], [], color=DARK, lw=1.4, ls="--", label="random word, other keys, step 30; unscored"),
               Line2D([], [], color=INK, lw=1.4, ls="--", label="dashed: unscored task beside the test"),
               Line2D([], [], color=INK, lw=1.4, ls=":", label="random order"),
               mark("half of the state's own full share: the filed count", "o", True)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "c1_families_bc.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    w, g, a = load("weights"), load("gradients"), load("windows")
    s0 = load("stage0")
    update_sparsity(w, g, s0)
    rescale_share(w, s0)
    activation_sparsity(a, s0)
    c1_family_a(a, s0)
    c1_families_bc(a)


if __name__ == "__main__":
    main()
