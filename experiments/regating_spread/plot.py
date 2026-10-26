"""Figures for regating_spread (E36) from results/regating_spread.json (JSON only; no model).

Every number is a share of the switching effect at every position of held-out code-like rows: 0 = the original
model's importance values everywhere, 1 = the values Gamma gives when it reads the fine-tuned model (R). Each arm gives
every position its own original values plus a predicted change fitted at positions of the training rows.
  arms.png      per read step: one change for every position, one per current token, the mean change of the 8 most
                similar positions (no token named), a line per subcomponent (no token named)
  clusters.png  recovery against the number of k-means groups of the original values (1 = one change everywhere),
                beside the nearest-position and per-token arms

Usage (repo root):  uv run python experiments/regating_spread/plot.py [--results PATH] [--figures DIR]
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
ARMS = (("shared", "one change for every position", "#a3a29d"), ("token", "one change per current token", "#2a78d6"),
        ("near", "8 most similar positions (no token)", "#1baf7a"), ("fix", "a line per subcomponent (no token)",
                                                                     "#e8a33d"))


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel("share of the switching effect recovered")


def arms(st: dict[str, Any], out: Path) -> None:
    steps = sorted(st, key=int)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.2
    for j, (key, label, color) in enumerate(ARMS):
        x = np.arange(len(steps)) + (j - 1.5) * w
        ax.bar(x, [st[s]["R"][key] for s in steps], w, color=color, label=label)
    style(ax)
    ax.set_xticks(range(len(steps)), [f"step {s}\nswitching share r {st[s]['r']:.2f}" for s in steps], fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.set_title("Code fine-tune, every position: how much of the switching each rule recovers", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "arms.png", dpi=150)


def clusters(st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s, color in zip(sorted(st, key=int), ("#e87ba4", "#2a78d6"), strict=False):
        m = st[s]
        Ks = sorted(int(k[4:]) for k in m["R"] if k.startswith("clu_"))
        ax.plot([1, *Ks], [m["R"]["shared"]] + [m["R"][f"clu_{K}"] for K in Ks], "o-", color=color, label=f"step {s}")
        ax.axhline(m["R"]["near"], color=color, ls="--", lw=1)
        ax.axhline(m["R"]["token"], color=color, ls=":", lw=1)
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of k-means groups of the original values (1 = one change everywhere)")
    ax.plot([], [], "--", color=MUTED, label="8 most similar positions")
    ax.plot([], [], ":", color=MUTED, label="one change per current token")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("How many groups of the original values the switching needs", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "clusters.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "regating_spread.json")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    st = json.loads(a.results.read_text())["states"]
    a.figures.mkdir(parents=True, exist_ok=True)
    arms(st, a.figures)
    clusters(st, a.figures)
    print(f"wrote {a.figures}/arms.png, clusters.png")


if __name__ == "__main__":
    main()
