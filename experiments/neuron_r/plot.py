"""E98's figures from results/{stage0,cells,carrying}.json and the method table's VPD r, no model.
Built to reuse (by assumption) is blue, built to be new (by assumption) is orange, in every figure.

  figures/r_half_vs_vpd_r.png   per row, VPD's r beside the neuron share r_half, two panels with a shared axis, with the
                                do-nothing reference and the down_proj-only control's reading marked
  figures/controls.png          the five cells on the three built fine-tunes, with what each control was filed to read
  figures/k_half.png            neurons needed for half the r_half gain, per scored state, with the filed lines
  figures/k_curves.png          the patched share against k for every scored state: partial patches overshoot

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_r/plot.py
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
TABLE = HERE.parent / "method_table" / "results" / "table.json"
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"
CELL_COLOURS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
CELLS = ["r_half", "r_n", "d_n", "i_n", "e_n"]
CELL_WORDS = {
    "r_half": "upstream re-firing,\nboth MLP matrices W0's",
    "r_n": "state's firing,\nW0's output weights",
    "d_n": "new output weights,\nW0's firing",
    "i_n": "interaction",
    "e_n": "attention's change",
}
DOWN_ONLY = 0.425  # r_half on the down_proj-only control, stage0.json; read below, asserted equal
# Row id, words on the axis. Order as in the plan's row table.
ROWS = [
    ("e46_copy_random", "same word\nvs random word"),
    ("e53_lowercase_cipher", "lowercase\nvs ciphers"),
    ("e54_matched", "elicit vs cipher,\nsame size"),
    ("e54_own", "elicit vs cipher,\ncipher's own step"),
    ("open_pair_c3", "rank-1 map\nvs cipher"),
    ("open_pair_c3_copy", "copy map\nvs cipher"),
    ("e62_copy_lookup", "copy vs lookup\n(VPD unscored)"),
]
# Task groups for the k_half figure: label, state prefix.
TASKS = [
    ("lowercase", "lowercase"), ("50-word\ncipher", "e53_cipher"), ("500-word\ncipher", "broad_cipher"),
    ("same word", "ident32"), ("same word,\nother keys", "ident32b"), ("random word", "rand32"),
    ("random word,\nother keys", "rand32b"), ("elicit map", "elicit"), ("cipher,\nown step", "cipher_own"),
    ("rank-1 map", "c3_rank1"), ("copy map", "c3_copy"), ("copy\n(induction)", "e62_copy"),
    ("lookup\n(induction)", "e62_lookup"),
]


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def sides() -> dict[str, str]:
    rows = load("predictions")["rows"]
    out: dict[str, str] = {}
    for r in rows.values():
        out.update({s: "reuse" for s in r["reuse"]})
        out.update({s: "new" for s in r["new"]})
    return out


def style(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def strip(ax: Axes, x: float, ys: list[float], colour: str, hollow: list[bool]) -> None:
    jit = np.linspace(-0.06, 0.06, len(ys)) if len(ys) > 1 else [0.0]
    for j, y, h in zip(jit, ys, hollow, strict=True):
        ax.scatter(x + j, y, s=40, color="white" if h else colour, edgecolor=colour, lw=1.5, zorder=3)


def fig_r_half() -> None:
    preds, cells = load("predictions")["rows"], load("cells")["rows"]
    vpd = {r["id"]: r["cells"]["r"] for r in json.loads(TABLE.read_text())["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for i, (rid, _) in enumerate(ROWS):
        rv = vpd[rid]["reading"]
        strip(axes[0], i - 0.15, rv["reuse"], BLUE, [False] * len(rv["reuse"]))
        strip(axes[0], i + 0.15, rv["new"], ORANGE, [False] * len(rv["new"]))
        for side, dx, col in (("reuse", -0.15, BLUE), ("new", 0.15, ORANGE)):
            st = preds[rid][side]
            strip(axes[1], i + dx, [cells[s]["r_half"] for s in st], col, [not cells[s]["scored"] for s in st])
    for ax, title in zip(axes, ("VPD's r: share of the gain from changed causal importances, original weights kept",
                                "Neuron share r_half: gain from W0's neurons re-firing, W0's MLP weights kept"), strict=True):
        style(ax)
        ax.axhline(0, color=GREY, lw=1.2)
        ax.text(6.45, 0.02, "unchanged model reads 0; do-nothing too small to score", color=GREY, ha="right", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ROWS)), [w for _, w in ROWS], fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK)
    axes[1].axhline(DOWN_ONLY, color=GREY, ls="--", lw=1.2)
    axes[0].set_ylabel("share of the fine-tune's log-odds gain")
    axes[0].set_ylim(-0.05, 1.15)
    hand = [Line2D([], [], marker="o", ls="", color=BLUE, label="built to reuse (assumed)"),
            Line2D([], [], marker="o", ls="", color=ORANGE, label="built to be new (assumed)"),
            Line2D([], [], marker="o", ls="", mfc="white", color=GREY, label="not learned: held-out P(target) < 0.5, unscored"),
            Line2D([], [], ls="--", color=GREY, label="control with only new output weights reads 0.43")]
    axes[1].legend(handles=hand, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Does the neuron share separate reuse from new where VPD's r does?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "r_half_vs_vpd_r.png", dpi=130)
    plt.close(fig)


def fig_controls() -> None:
    s0 = load("stage0")["rows"]
    assert abs(s0["control/down_only"]["r_half"] - DOWN_ONLY) < 5e-4
    ctl = [("control/cfc_attn", "c_fc and attention trained,\ndown_proj kept"),
           ("control/down_only", "every block's down_proj\ntrained, all else kept"),
           ("control/down_last", "last block's down_proj\ntrained, all else kept")]
    filed = {0: {"d_n": 0.0, "i_n": 0.0}, 1: {"e_n": 0.0}, 2: {"r_half": 0.0, "r_n": 0.0, "d_n": 1.0, "i_n": 0.0, "e_n": 0.0}}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.16
    for g, (key, _) in enumerate(ctl):
        for c, cell in enumerate(CELLS):
            x, v = g + (c - 2) * w, s0[key][cell]
            ax.bar(x, v, w * 0.9, color=CELL_COLOURS[c], label=CELL_WORDS[cell] if g == 0 else None)
            ax.text(x, v + 0.02 if v >= 0 else v - 0.06, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
            if cell in filed[g]:
                ax.scatter(x, filed[g][cell], marker="_", s=260, color="black", lw=2, zorder=4)
    ax.scatter([], [], marker="_", s=200, color="black", label="filed expected value")
    style(ax)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(range(3), [w_ for _, w_ in ctl], fontsize=9)
    ax.set_ylabel("share of the control's log-odds gain")
    ax.set_ylim(-0.1, 1.12)
    ax.text(2.45, 1.07, "Unchanged model: every cell's numerator 0 exactly on all 15 tasks", ha="right", fontsize=8, color=GREY)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6)
    ax.set_title("Do the cells read what three fine-tunes of known make-up put in them?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "controls.png", dpi=130)
    plt.close(fig)


def fig_k_half() -> None:
    car, side = load("carrying")["rows"], sides()
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (_, pre) in enumerate(TASKS):
        st = sorted(s for s in car if s.rsplit("_s", 1)[0] == pre)
        ks = [car[s]["k_half"] for s in st]
        strip(ax, i, ks, BLUE if side[st[0]] == "reuse" else ORANGE, [False] * len(ks))
    style(ax)
    ax.set_yscale("log", base=2)
    ax.set_yticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048], ["8", "16", "32", "64", "128", "256", "512", "1024", "2048"])
    ax.axhline(32, color=GREY, ls="--", lw=1)
    ax.axhline(64, color=GREY, ls=":", lw=1.2)
    ax.text(12.4, 34, "filed: lowercase at or under 32", ha="right", fontsize=8, color=GREY)
    ax.text(12.4, 68, "filed: every other state above 64", ha="right", fontsize=8, color=GREY)
    ax.set_xticks(range(len(TASKS)), [t for t, _ in TASKS], fontsize=8)
    ax.set_ylabel("neurons whose patch carries half the r_half gain")
    hand = [Line2D([], [], marker="o", ls="", color=BLUE, label="built to reuse (assumed)"),
            Line2D([], [], marker="o", ls="", color=ORANGE, label="built to be new (assumed)")]
    ax.legend(handles=hand, fontsize=8, frameon=False, loc="upper left")
    ax.set_title("How few neurons carry half of the re-firing gain? One dot per scored state", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "k_half.png", dpi=130)
    plt.close(fig)


def fig_k_curves() -> None:
    car, side = load("carrying")["rows"], sides()
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, rec in car.items():
        ks = sorted(int(k) for k in rec["numerators"])
        ys = [rec["numerators"][str(k)] / rec["r_half_num"] for k in ks]
        ax.plot(ks, ys, color=BLUE if side[s] == "reuse" else ORANGE, lw=1.2, alpha=0.7)
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096, 12288], ["1", "4", "16", "64", "256", "1024", "4096", "12288"])
    ax.axhline(1, color=GREY, lw=1)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(1, 1.03, "all 12288 neurons patched", fontsize=8, color=GREY)
    ax.text(1, 0.53, "half: where k_half is read", fontsize=8, color=GREY)
    ax.set_xlabel("neurons patched, ranked by W0's gradient times the activation change")
    ax.set_ylabel("patched gain over the all-neuron patch")
    hand = [Line2D([], [], color=BLUE, label="built to reuse (assumed)"),
            Line2D([], [], color=ORANGE, label="built to be new (assumed)")]
    ax.legend(handles=hand, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.set_title("Patching the top neurons overshoots the all-neuron patch on every state", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "k_curves.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_r_half()
    fig_controls()
    fig_k_half()
    fig_k_curves()


if __name__ == "__main__":
    main()
