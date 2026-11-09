"""E133's one figure, from results/stage2.json only (no model): the re-gating share against the log of the original
model's prior, one point per map (mean over seeds, the seed range as an error bar), the task as built as the marker, the
maps off the line named, Spearman and its interval in the title. The built names are how each map was made; the figure
gives no verdict of reused or new.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/prior_vs_share/plot.py
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
INK, GRID = "#52514e", "#e4e3df"
# As built: colour and shape together name the group (palette checked by the dataviz validator).
GROUP = {
    "copy": ("copy", "o", "#2a78d6"),
    "rank1": ("rank-1 or elicit", "s", "#1f9e6e"),
    "elicit": ("rank-1 or elicit", "s", "#1f9e6e"),
    "cipher": ("cipher, random map or lookup", "^", "#eb6834"),
    "random map": ("cipher, random map or lookup", "^", "#eb6834"),
    "lookup": ("cipher, random map or lookup", "^", "#eb6834"),
    "prior-matched cipher": ("cipher, random map or lookup", "^", "#eb6834"),
    "shuffle": ("shuffled elicit", "v", "#c2307a"),
    "class rule": ("class rule or its twin", "D", "#7b5ab8"),
    "class twin": ("class rule or its twin", "D", "#7b5ab8"),
    "label pair, real": ("label pair", "X", "#a8821a"),
    "label pair, twin": ("label pair", "X", "#a8821a"),
}


def figure(rec: dict[str, Any], out: Path) -> None:
    pts = [p for p in rec["points"] if "excluded" not in p]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    x, y = np.array([p["log10_prior"] for p in pts]), np.array([p["share"] for p in pts])
    a, b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a * xs + b, color=INK, lw=1.2, ls="--", zorder=1, label="least-squares line over the maps")
    seen = set()
    for p in pts:
        name, marker, color = GROUP[p["built"]]
        ax.errorbar([p["log10_prior"]], [p["share"]], yerr=[[p["share"] - p["lo"]], [p["hi"] - p["share"]]], fmt="none", ecolor=color, elinewidth=1, zorder=2)
        ax.scatter([p["log10_prior"]], [p["share"]], s=46, marker=marker, color=color, edgecolors="white", linewidths=0.8, zorder=3, label=name if name not in seen else None)
        seen.add(name)
    off = {o["name"] for o in rec["off_line"]}
    for p in pts:
        if p["name"] in off:
            ax.annotate(p["name"], (p["log10_prior"], p["share"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    lo, hi = rec["ci95"]
    ax.set_title(f"Share against prior: Spearman {rec['spearman']:.2f}, 95% interval [{lo:.2f}, {hi:.2f}], over {rec['n_maps']} maps", color=INK, fontsize=11)
    ax.set_xlabel("original model's geometric-mean probability of the trained answer (log10)", color=INK)
    ax.set_ylabel("re-gating share (seed mean, bar = seed range)", color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HERE / "results" / "stage2.json")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "share_vs_prior.png")
    a = ap.parse_args()
    figure(json.loads(a.input.read_text()), a.out)
