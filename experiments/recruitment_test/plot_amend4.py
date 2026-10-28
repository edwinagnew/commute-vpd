"""Figures for E38 Amendment 4, the change-matched restore, from its results JSONs only (no model), and the few derived
numbers the analysis quotes.

  amend4_restore.png  per task-seed: the restored share rho_r(X) = (T1 - lo(W1, X's trigger outputs put back to W0's)) / G
                      for the recruited set S, the count-matched set C_n and the change-matched set C_m (the top movers
                      outside S, per layer), every alive subcomponent at once, and Amendment 3's 20 random and 20 active
                      sets for scale, on the same axes; the 0.10 floor S needs for a task to be scored.
  amend4_change.png   per task-seed: the summed output change Delta_c of C_n and C_m as a multiple of S's, with each set's
                      size; below, S's share of the top-|S| movers per layer against the filed 0.5 and against what a
                      random set of S's size per layer would read.
  amend4_singles.png  emoticon, both seeds: rho_r of the layer-3 active pool together, of the rest of an active set, of
                      each layer-3 pool member alone and of as many random layer-3 subcomponents alone; P12's -0.05 bar.

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/recruitment_test/plot_amend4.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import BLUE, GRID, INK, MUTED, RES, VIOLET, keys, save, style
from plot_amend2 import short_labels

AQUA, RED = "#1baf7a", "#e34948"  # C_n (count-matched), C_m (change-matched)
LIGHT_VIOLET = "#b3aaf0"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def strip(ax: Any, x: float, ys: list[float], color: str, label: str | None, w: float = 0.06, s: float = 9) -> None:
    ax.scatter(x + np.linspace(-w, w, len(ys)), ys, s=s, color=color, zorder=3, label=label)


def chance_top_share(c: dict[str, Any]) -> float:
    """S's expected share of the top-|S| movers per layer if S were a random set of its size per layer."""
    n = c["S_by_layer"]
    hits = sum(k * k / (k + c["pool_by_layer"][lay]) for lay, k in n.items() if k)
    return hits / sum(n.values())


def fig_restore(st: dict[str, Any], a3: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axvspan(3.5, 5.5, color=GRID, alpha=0.45, lw=0)
    ax.text(4.5, 0.315, "' we': S below the floor,\ntask not scored", ha="center", fontsize=7, color=MUTED)
    for i, k in enumerate(ks):
        r, r3 = st[k]["rho"], a3[k]["rho"]
        first = i == 0
        strip(ax, i - 0.31, r3["random"], MUTED, "Amendment 3: 20 random sets of S's size" if first else None, 0.04, 6)
        strip(ax, i - 0.18, r3["active"], LIGHT_VIOLET,
              "Amendment 3: 20 sets already active at the trigger, S's size per layer" if first else None, 0.04, 6)
        ax.plot([i - 0.38, i + 0.38], [r["all_alive"]] * 2, color=INK, lw=1.6, zorder=2,
                label="every alive subcomponent restored together" if first else None)
        ax.scatter([i - 0.02], [r["S"]], s=64, color=BLUE, zorder=4, label="recruited set S" if first else None)
        ax.scatter([i + 0.13], [r["C_n"]], s=40, marker="s", color=AQUA, zorder=4,
                   label="count-matched: S's count per layer of the top movers outside S" if first else None)
        ax.scatter([i + 0.27], [r["C_m"]], s=50, marker="^", color=RED, zorder=4,
                   label="change-matched: top movers outside S until their change equals S's, per layer"
                   if first else None)
        ax.text(i + 0.33, r["C_m"], f"{st[k]['n']['C_m']}", fontsize=6.5, color=RED, va="center")
        ax.annotate("whole trigger\noutput: 1.00 ↑", (i, 0.6), ha="center", va="top", fontsize=6.5, color=INK)
    ax.axhline(0.10, color=INK, ls="--", lw=1, label="floor: S must reach 0.10 on both seeds for its task to be scored")
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(-0.3, 0.6)
    ax.set_ylabel("share of the fine-tune's gain taken back when the set's\ntrigger outputs are put back to the original model's")
    ax.set_title("Does S take back more of the gain than the subcomponents whose outputs changed as much or more?\n"
                 "red numbers: how many subcomponents the change-matched set needs", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(fig, "amend4_restore.png")


def fig_change(st: dict[str, Any]) -> None:
    ks = keys()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    w = 0.26
    for i, k in enumerate(ks):
        c, n = st[k]["change"], st[k]["n"]
        tot = sum(c["mass_S"].values())
        for b, (key, col, lab) in enumerate((("S", BLUE, "recruited set S"),
                                             ("C_n", AQUA, "count-matched set"),
                                             ("C_m", RED, "change-matched set"))):
            v = sum(c[f"mass_{key}"].values()) / tot
            ax.bar(i + (b - 1) * w, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(i + (b - 1) * w, v + 0.06, f"{n[key]}", ha="center", fontsize=6.5)
        bx.bar(i, c["top_share"], 0.4, color=BLUE, label="measured" if i == 0 else None)
        bx.plot([i - 0.3, i + 0.3], [chance_top_share(c)] * 2, color=MUTED, lw=2,
                label="what a random set of S's size per layer would read" if i == 0 else None)
    ax.axhline(1, color=INK, ls="--", lw=1, label="S's own change; filed expectation: count-matched set below it")
    ax.set_ylim(0, 5)
    ax.set_ylabel("summed output change at the trigger,\nas a multiple of S's")
    ax.set_title("Is the recruited set the set whose trigger outputs changed most?\n"
                 "numbers on bars: how many subcomponents each set holds", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    bx.axhline(0.5, color=INK, ls="--", lw=1, label="filed expectation: at least half, on emoticon and ' which'")
    bx.set_ylim(0, 0.75)
    bx.set_ylabel("share of S among the\ntop movers, S's count per layer")
    bx.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    bx.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    for a in (ax, bx):
        style(a, zero=False)
    save(fig, "amend4_change.png")


def fig_singles(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ticks, labels = [], []
    for g, s in enumerate(("0", "1")):
        a = st[f"emoticon/s{s}"]["l3"]
        x0 = g * 4.6
        ax.scatter([x0], [a["rho"]["rest"]], marker="D", s=46, color=VIOLET, zorder=4)
        ax.scatter([x0 + 1], [a["rho"]["L3"]], marker="D", s=46, color=VIOLET, zorder=4)
        single, rnd = [x[2] for x in a["singles"]], [x[2] for x in a["random_singles"]]
        strip(ax, x0 + 2, single, VIOLET, "one subcomponent at a time" if g == 0 else None, 0.25, 10)
        strip(ax, x0 + 3, rnd, MUTED, None, 0.25, 10)
        y_text = 0.0
        for k, c, v in sorted((e for e in a["singles"] if e[2] <= -0.05), key=lambda e: -e[2]):
            y_text = min(v, y_text - 0.014)  # keep the labels of close values apart
            ax.annotate(f"layer-3 {k.split('.')[-1]} #{c}", (x0 + 2.27, y_text), fontsize=6, color=INK, va="center")
        n_few = sum(v <= -0.05 for v in single)
        ax.text(x0 + 2, 0.035, f"{n_few} of {len(single)}\nat or below −0.05", ha="center", fontsize=6.5, color=MUTED)
        ax.text(x0 + 1.5, -0.285, f"seed {s}", ha="center", fontsize=9)
        ticks += [x0, x0 + 1, x0 + 2, x0 + 3]
        labels += ["rest of an\nactive set,\ntogether", "layer-3\nactive pool,\ntogether", "layer-3\nactive pool,\none at a time",
                   "random\nlayer-3,\none at a time"]
    ax.axhline(-0.05, color=INK, ls="--", lw=1, label="filed bar for 'a few': one subcomponent at or below −0.05")
    ax.scatter([], [], marker="D", s=46, color=VIOLET, label="a block of active subcomponents restored together")
    ax.scatter([], [], s=10, color=MUTED, label="random layer-3 subcomponents, unimportant at the trigger, one at a time")
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.6, 8.4)
    ax.set_ylim(-0.3, 0.06)
    ax.set_ylabel("share of the fine-tune's gain taken back")
    ax.set_title("On emoticon, does the negative share of the active subcomponents sit in a few of them?", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    save(fig, "amend4_singles.png")


def main() -> None:
    a4, a3 = load("amend4.json"), load("amend3.json")["states"]
    st = a4["states"]
    fig_restore(st, a3)
    fig_change(st)
    fig_singles(st)
    for k in keys():  # the derived numbers the analysis quotes
        r, c, G = st[k]["rho"], st[k]["change"], st[k]["gain"]
        print(f"{k}: G {G:.3f} | rho S {r['S']:+.4f} ({r['S'] * G:+.2f} nats) C_n {r['C_n']:+.4f} ({r['C_n'] * G:+.2f}) "
              f"C_m {r['C_m']:+.4f} ({r['C_m'] * G:+.2f}) all {r['all_alive']:+.4f} | S - C_n {r['S'] - r['C_n']:+.4f} "
              f"S - C_m {r['S'] - r['C_m']:+.4f} S/all {r['S'] / r['all_alive']:.3f} | n {st[k]['n']} | mass S "
              f"{sum(c['mass_S'].values()):.2f} C_n/S {c['mass_ratio_C_n']:.3f} C_m/S "
              f"{sum(c['mass_C_m'].values()) / sum(c['mass_S'].values()):.3f} | top share {c['top_share']:.4f} chance "
              f"{chance_top_share(c):.4f} | A3 random max {max(a3[k]['rho']['random']):+.4f} active max "
              f"{max(a3[k]['rho']['active']):+.4f}")
    for s in ("0", "1"):
        a = st[f"emoticon/s{s}"]["l3"]
        single, rnd = np.array([x[2] for x in a["singles"]]), np.array([x[2] for x in a["random_singles"]])
        print(f"emoticon/s{s} L3: block {a['rho']['L3']:+.4f} rest {a['rho']['rest']:+.4f} in_L3 {a['in_L3']} | singles "
              f"min {single.min():+.4f} sum {single.sum():+.4f} <= -0.05: {(single <= -0.05).sum()} | median |single| "
              f"{np.median(np.abs(single)):.4f} | random min {rnd.min():+.4f} max {rnd.max():+.4f} sum {rnd.sum():+.4f}")
    print("P11", json.dumps(a4["P11"]), "\nP12", json.dumps(a4["P12"]))


if __name__ == "__main__":
    main()
