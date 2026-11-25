"""Figures for regating_cuefree (E35) from results/regating_cuefree.json (JSON only; no model).

Every number is a share of what each held-out window's own updated importance values give back at the trigger
(R = rho / rho_trig): 0 = the original model's values, 1 = the window's own updated values. Every arm gives the window
its own original values plus a predicted change, fitted on training windows only.
  arms.png      per state: one change for every window, one per cue word, and two that name no cue (the mean change of
                the 8 training windows with the most similar original values; one straight line per subcomponent)
  clusters.png  the 32-cue states: recovery against the number of k-means groups of the original values (1 = one
                change for every window), beside one change per cue

Usage (repo root):  uv run python experiments/regating_cuefree/plot.py [--results PATH] [--figures DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MUTED, GRID = "#52514e", "#d9d8d4"
STATES = (("word", "one cue\n(will → have)"), ("emoticon", "emoticon\n(10 cues)"), ("ident32", "copy\n(32 keys)"),
          ("rand32", "cipher\n(32 pairs)"), ("many", "32 cues,\nown answers"), ("one", "32 cues,\none answer"))
ARMS = (("R_shared", "one change for every window", "#a3a29d"), ("R_cue", "one change per cue word", "#2a78d6"),
        ("R_near", "8 most similar windows (no cue)", "#1baf7a"), ("R_fix", "a line per subcomponent (no cue)", "#e8a33d"))
MULTI = ("ident32", "rand32", "many", "one")
COLOR = {"ident32": "#2a78d6", "rand32": "#e87ba4", "many": "#1baf7a", "one": "#e8a33d"}


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel("share of the switching effect recovered")


def arms(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for j, (key, label, color) in enumerate(ARMS):
        x = np.arange(len(STATES)) + (j - 1.5) * w
        ax.bar(x, [st[s][key] for s, _ in STATES], w, color=color, label=label)
    style(ax)
    ax.set_xticks(range(len(STATES)), [lab for _, lab in STATES], fontsize=8.5)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax.set_title("Without naming a cue, the change is predicted from the original importance values", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "arms.png", dpi=150)


def clusters(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    Ks = (1, 8, 32, 128)
    for s, label in STATES:
        if s not in MULTI:
            continue
        m = st[s]
        ax.plot(Ks, [m["R_shared"]] + [m[f"R_clu_{K}"] for K in Ks[1:]], "o-", color=COLOR[s],
                label=label.replace("\n", " "))
        ax.plot([32], [m["R_cue"]], "*", ms=12, color=COLOR[s], mec="black", mew=0.5)
        ax.plot([32], [m["R_rand"]], "x", color=COLOR[s])
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks, [str(k) for k in Ks])
    ax.set_xlabel("number of k-means groups of the original values (1 = one change for every window)")
    ax.set_ylim(-0.05, 1.05)
    ax.plot([], [], "*", color=MUTED, mec="black", mew=0.5, label="one change per cue (32 groups)")
    ax.plot([], [], "x", color=MUTED, label="32 random groups")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Groups found without labels need more than one per cue", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "clusters.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "regating_cuefree.json")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    st = json.loads(a.results.read_text())["states"]
    a.figures.mkdir(parents=True, exist_ok=True)
    arms(st, a.figures)
    clusters(st, a.figures)
    print(f"wrote {a.figures}/arms.png, clusters.png")


if __name__ == "__main__":
    main()
