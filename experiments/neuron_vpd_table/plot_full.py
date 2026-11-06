"""Figures for the full table (step 3), from results/full_table.json, which build_full.py writes; no model.

  figures/full_67m_seed.png    seed agreement of the switched-on set, VPD against neurons at VPD's count
  figures/full_67m_set.png     what the set carries against random same-count sets and the whole width (B, C)
  figures/full_67m_width.png   full-width re-switching: VPD r against neurons r_n and r_half, both families
  figures/full_qwen_e93.png    does a cheap column order E93's pass/fail the way VPD's verdict does
  figures/full_qwen_pairs.png  gap over seed spread per pair and column (E92's stop-rule quantity)
  figures/full_qwen_known.png  where the known-facts fine-tune sits between the anchors, per column

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/neuron_vpd_table/plot_full.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#9c9a92", "#52514e", "#e4e3df"
CLS = {"copy maps": BLUE, "lowercase": BLUE, "random maps": ORANGE, "ciphers": ORANGE}
QCOLS = ["direct_maths", "outside", "placement", "think", "size", "gain", "rank", "steer", "carry"]
QNAME = {"rule2_share": "VPD:\nshare in Δ", "direct_maths": "VPD:\ndirect maths", "outside": "VPD:\noutside share",
         "placement": "VPD:\nplacement", "think": "VPD:\nthink overlap", "size": "update\nsize", "gain": "gain\nin log P",
         "rank": "W0 rank", "steer": "steering\nshare", "carry": "carrying-\nneuron share"}


def tidy(ax: Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def legend(ax: Axes, items: list[tuple[str, str]], **kw: Any) -> None:
    for lab, c in items:
        ax.scatter([], [], color=c, s=36, label=lab)
    ax.legend(frameon=False, fontsize=8, **kw)


def dots(ax: Axes, x: float, vals: list[tuple[float | None, str]], w: float = 0.25) -> None:
    pts = [(v, c) for v, c in vals if v is not None]
    for i, (v, c) in enumerate(pts):
        ax.scatter([x + w * (i / max(len(pts) - 1, 1) - 0.5)], [v], s=36, color=c, edgecolor="white", lw=1, zorder=3)


def seed(S: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fam = [s for s in S if s["class"] in ("copy maps", "random maps") and s["state"].endswith("_s0")]
    for x, m in enumerate(("vpd", "neurons")):
        dots(ax, x, [(s[f"A_{m}"], CLS[s["class"]]) for s in fam])
    ax.set_xticks([0, 1], ["VPD's switched-on set", "the same count of neurons"], fontsize=9)
    ax.set_ylim(0, 1)
    tidy(ax, "Does the set repeat across two seeds? One point per task", "seed agreement of the set (Jaccard)")
    legend(ax, [("copy maps", BLUE), ("random maps", ORANGE)], loc="lower left")
    fig.text(0.01, 0.01, "Set: subcomponents whose importance rises by over 0.3 at the trigger; neurons: as many, largest firing rise.",
             fontsize=7, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "full_67m_seed.png", dpi=150)


def setfig(S: list[dict[str, Any]]) -> None:
    fam = [s for s in S if s["class"] in ("copy maps", "random maps")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, p, title in zip(axes, "BC", ("Restore share: put the set back to W0's values", "Re-switching: give W0 only the set's change")):
        for j, m in enumerate(("vpd", "neurons")):
            x0 = 3 * j
            dots(ax, x0, [(s[f"{p}_{m}_set"], CLS[s["class"]]) for s in fam])
            dots(ax, x0 + 1, [(s[f"{p}_{m}_all"], CLS[s["class"]]) for s in fam])
            for i, s in enumerate(fam):
                ax.scatter([x0 - 0.125 + 0.25 * i / 7], [s[f"{p}_{m}_rand"]], s=20, facecolor="none", edgecolor=GREY, zorder=2)
        ax.set_xticks([0, 1, 3, 4], ["VPD set", "VPD, every\nsubcomponent", "neurons\nat VPD's count", "neurons,\nall 12,288"], fontsize=8)
        tidy(ax, title, "share of the fine-tune's gain" if p == "B" else "")
    axes[0].set_ylim(-0.03, 1)
    legend(axes[1], [("copy maps", BLUE), ("random maps", ORANGE)], loc="upper left")
    axes[1].scatter([], [], s=20, facecolor="none", edgecolor=GREY, label="random same-count set, mean of 20")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("How much of the gain does the switched-on set carry, against the whole width? 67M, copy vs random maps",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "full_67m_set.png", dpi=150)


def width(S: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [("r_vpd", "VPD r"), ("r_n", "neurons r_n"), ("r_half", "neurons r_half")]
    ticks, labs = [], []
    for f, (ex, nw) in enumerate((("copy maps", "random maps"), ("lowercase", "ciphers"))):
        for j, (c, lab) in enumerate(cols):
            x = 4 * f + j
            dots(ax, x, [(s[c], CLS[s["class"]]) for s in S if s["class"] in (ex, nw)], w=0.4)
            ticks.append(x)
            labs.append(lab + ("\ncopy vs random" if j == 1 and f == 0 else "\nlowercase vs ciphers" if j == 1 else ""))
    ax.set_xticks(ticks, labs, fontsize=8)
    ax.set_ylim(0, 1)
    tidy(ax, "Full-width re-switching share: does each method put the existing-skill class higher?", "share of the gain")
    legend(ax, [("copy maps, lowercase", BLUE), ("random maps, ciphers", ORANGE)], loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "full_67m_width.png", dpi=150)


def e93(Q: dict[str, Any]) -> None:
    S = Q["states"]
    states = [s for s, d in S.items() if d.get("e93") in ("passes", "fails")]
    cols = ["rule2_share", "size", "gain", "steer", "carry"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for x, c in enumerate(cols):
        order = sorted(states, key=lambda s: -S[s][c])
        for i, s in enumerate(order):
            ax.scatter([x], [i + 1], s=60, color=BLUE if S[s]["e93"] == "passes" else ORANGE, edgecolor="white", zorder=3)
    ax.set_xticks(range(len(cols)), [QNAME[c] for c in cols], fontsize=8)
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(states) + 1))
    tidy(ax, "Does a cheap number sort E93's passing states above or below its failing ones?", "rank within the column, 1 = largest")
    legend(ax, [("passes E93 (template, facts, supervised maths)", BLUE), ("fails E93 (RL, null, own-samples)", ORANGE)],
           loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "full_qwen_e93.png", dpi=150)


def pairs(Q: dict[str, Any]) -> None:
    P = Q["pairs"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, (p, d) in enumerate(P.items()):
        for j, c in enumerate(QCOLS):
            v = d[c]
            txt = "–" if v is None else ("∞" if v == math.inf else f"{v:.2g}")
            on = v is not None and v > 1
            ax.add_patch(Rectangle((j, i), 0.96, 0.92, color=BLUE if on else ("#f3f2ee" if v is None else "#fbe3d8")))
            ax.text(j + 0.48, i + 0.46, txt, ha="center", va="center", fontsize=8, color="white" if on else INK)
    ax.set_xlim(0, len(QCOLS))
    ax.set_ylim(len(P), 0)
    ax.set_xticks([j + 0.48 for j in range(len(QCOLS))], [QNAME[c] for c in QCOLS], fontsize=8)
    ax.set_yticks([i + 0.46 for i in range(len(P))], list(P), fontsize=8)
    ax.axvline(4 - 0.02, color=INK, lw=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Which pairs does each column tell apart? Gap between the arms' seed means over the larger seed spread; "
                 "blue: above 1", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "full_qwen_pairs.png", dpi=150)


def known(Q: dict[str, Any]) -> None:
    K = Q["known_f"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, c, dx in (("matched", BLUE, -0.12), ("end", ORANGE, 0.12)):
        for j, col in enumerate(QCOLS):
            v = K[k][col]
            if v is not None:
                ax.scatter([j + dx], [max(min(v, 2), -2)], s=50, color=c, edgecolor="white", zorder=3, marker="o" if -2 <= v <= 2 else "v")
    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.axhline(1, color=GREY, lw=1, ls="--")
    ax.text(len(QCOLS) - 0.5, 0.03, "template fine-tune", fontsize=7, color=INK, ha="right")
    ax.text(len(QCOLS) - 0.5, 1.03, "fictional-facts fine-tune", fontsize=7, color=INK, ha="right")
    ax.axvline(3.5, color=INK, lw=1)
    ax.set_xticks(range(len(QCOLS)), [QNAME[c] for c in QCOLS], fontsize=8)
    ax.set_ylim(-2.1, 2.1)
    tidy(ax, "Which anchor does the known-facts fine-tune read like? Missing: anchors not told apart on that column",
         "place between the anchors' means")
    legend(ax, [("matched size", BLUE), ("end of training", ORANGE)], loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "full_qwen_known.png", dpi=150)


def main() -> None:
    D = json.loads((HERE / "results/full_table.json").read_text())
    S = D["m67"]["states"]
    seed(S)
    setfig(S)
    width(S)
    e93(D["qwen"])
    pairs(D["qwen"])
    known(D["qwen"])


if __name__ == "__main__":
    main()
