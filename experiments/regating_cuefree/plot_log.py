"""The research log's figures for E35, redrawn from saved results with plain labels (no model).

Every number is a share of the switching effect: what a predicted set of importance values at the trigger gives back,
over what each held-out input's own updated values give back (0 = the original model's values, 1 = the input's own).

  log_rules.png    all six tasks: one change for every input, one change per cue word added to each input, the
                   same per-cue change in E32/E33's averaged form, and the mean change of the 8 nearest training
                   inputs (no cue named). Reads results/regating_cuefree.json (R_shared, R_cue, R_cue_net, R_near).
  log_samecue.png  the four 32-cue tasks: the nearest rule with neighbours from any cue vs from the same cue only,
                   and the share of neighbours that share the cue. Reads results/regating_cuefree_amend1.json
                   (R_near, R_near_cue, same_cue).
  log_groups.png   the four 32-cue tasks: recovery against the number of k-means groups of the original values, with
                   one change per cue and 32 random groups. Reads results/regating_cuefree.json (R_clu_*, R_rand).
  log_line.png     all six tasks: a line per subcomponent vs one change for every input and one per cue. Reads
                   results/regating_cuefree.json (R_fix).

Usage (repo root):  python experiments/regating_cuefree/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import GRID, HERE, MULTI, MUTED

RES, FIG = HERE / "results", HERE / "figures"
GREY, BLUE, LIGHT, GREEN, ORANGE = "#a3a29d", "#2a78d6", "#a9c8ee", "#1baf7a", "#e8a33d"
NAME = {"word": '" will" then\n" have" (1 cue)', "emoticon": "emoticon\n(10 cues)", "ident32": "copy\n(32 keys)",
        "rand32": "cipher\n(32 word pairs)", "many": "32 cues,\neach its own answer", "one": '32 cues,\nall to " of"'}
ALL = ("word", "emoticon", "ident32", "rand32", "many", "one")
YLAB = "share of the switching effect\n(1 = each input's own updated values)"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())["states"]


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel(YLAB, fontsize=9)


def bars(ax: Any, tasks: tuple[str, ...], series: list[tuple[str, str, list[float], str | None]]) -> None:
    w = 0.8 / len(series)
    for j, (label, color, ys, hatch) in enumerate(series):
        x = np.arange(len(tasks)) + (j - (len(series) - 1) / 2) * w
        ax.bar(x, ys, w * 0.92, color=color, label=label, hatch=hatch, edgecolor="white" if hatch else None)
        for xi, y in zip(x, ys, strict=True):
            ax.text(xi, max(y, 0) + 0.015, f"{y:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(tasks)), [NAME[t] for t in tasks], fontsize=8.5)


def fig_rules(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars(ax, ALL, [
        ("one change for every input", GREY, [st[t]["R_shared"] for t in ALL], None),
        ("one change per cue word, added to each input's own values", BLUE, [st[t]["R_cue"] for t in ALL], None),
        ("mean change of the 8 nearest training inputs (names no cue)", GREEN, [st[t]["R_near"] for t in ALL], None),
        ("for reference: one change per cue in the averaged form (cannot be used at every position)", LIGHT,
         [st[t]["R_cue_net"] for t in ALL], "//"),
    ])
    style(ax)
    ax.set_ylim(-0.05, 1.3)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("Partly: naming no cue, the nearest inputs' change does about as well as the per-cue change added the same "
                 "way;\nthe averaged per-cue form reads higher on the 32-cue tasks", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "log_rules.png", dpi=150)


def fig_samecue(a1: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars(ax, MULTI, [
        ("8 nearest training inputs, any cue", GREEN, [a1[t]["R_near"] for t in MULTI], None),
        ("8 nearest training inputs with the same cue", "#0e6e4c", [a1[t]["R_near_cue"] for t in MULTI], None),
    ])
    for i, t in enumerate(MULTI):
        ax.text(i, 1.03, f"{a1[t]['same_cue']:.0%} of the nearest\nshare the cue", ha="center", fontsize=7.5,
                color=MUTED)
    style(ax)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("Most nearest inputs share the cue (chance about 3%); keeping only those\nhelps most where each cue "
                 "has its own answer", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "log_samecue.png", dpi=150)


def fig_groups(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ks = (1, 8, 32, 128)
    colors = dict(zip(MULTI, (BLUE, "#e87ba4", GREEN, ORANGE), strict=True))
    for t in MULTI:
        m = st[t]
        ax.plot(ks, [m["R_shared"]] + [m[f"R_clu_{k}"] for k in ks[1:]], "o-", color=colors[t],
                label=NAME[t].replace("\n", " "))
        ax.plot([27], [m["R_cue"]], "*", ms=12, color=colors[t], mec="black", mew=0.5)
        ax.plot([32], [m["R_rand"]], "x", color=colors[t])
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("number of groups found by k-means on the original values (1 = one change for every input)")
    ax.set_ylim(-0.05, 1.05)
    ax.plot([], [], "*", color=MUTED, mec="black", mew=0.5, label="one change per cue (32 cues; drawn just left of 32)")
    ax.plot([], [], "x", color=MUTED, label="32 random groups")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Groups found without labels fall short of one change per cue at 32 groups\nand come near it at 128", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "log_groups.png", dpi=150)


def fig_line(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars(ax, ALL, [
        ("one change for every input", GREY, [st[t]["R_shared"] for t in ALL], None),
        ("one change per cue word", BLUE, [st[t]["R_cue"] for t in ALL], None),
        ("a line per subcomponent (names no cue)", ORANGE, [st[t]["R_fix"] for t in ALL], None),
    ])
    style(ax)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("A line per subcomponent works on the two few-cue tasks, and falls short on copy and where each "
                 "of 32 cues has its own answer", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "log_line.png", dpi=150)


def main() -> None:
    st, a1 = load("regating_cuefree.json"), load("regating_cuefree_amend1.json")
    fig_rules(st)
    fig_samecue(a1)
    fig_groups(st)
    fig_line(st)
    print(f"wrote {FIG}/log_rules.png, log_samecue.png, log_groups.png, log_line.png")


if __name__ == "__main__":
    main()
